# Guide de réimplémentation : STFT + classifieur INT8 (CMSIS-DSP/CMSIS-NN) sur AudioMoth

Ce document rassemble tout ce qu'il faut savoir pour ré-implémenter, seul et en comprenant
chaque brique, le pipeline actuel : capture audio -> STFT -> normalisation (EMA ou PCEN,
voir §4/§6 - **PCEN est le choix actuellement déployé**) -> modèle `TinyLSTMCNN` quantifié en
INT8 -> classification, sur le AudioMoth Dev (EFM32WG380F256, Cortex-M4F @ 48MHz, 256KB flash,
32KB RAM interne + 256KB SRAM externe), dans l'un de trois modes firmware mutuellement
exclusifs (`STFT_DEBUG_MODE`, `TRIGGERED_RECORDING_MODE` - le mode de déploiement réel, voir
§15 -, ou le firmware normal) - voir `src/main.c`, tout en haut du fichier.

Il part du principe que vous ne connaissez pas CMSIS-DSP/CMSIS-NN, et détaille en particulier
les optimisations de convolution et les problèmes rencontrés avec la couche LSTM, qui ont été
longtemps la source du bug le plus sérieux de toute l'investigation - jusqu'à ce qu'un bug de
parité entraînement/déploiement encore plus important (§14.5) soit trouvé plus tard.

Sources citées : `src/main.c`, `src/model_inference.c` (ce dépôt), `export_int8.py`,
`export_pcen.py`, `compare_device_output.py`, `int8_sim.py`, `student_model.py`,
`train_windowed.py`, `dataset.py` (`test-audiomoth-cnn`), et le code source de CMSIS-NN
lui-même (`AudioMoth-Project/CMSIS-NN/Source/...`). Pour l'historique chronologique complet des
expériences (pas seulement le guide de réimplémentation), voir `EXPERIMENTS.md`.

---

## 1. Vue d'ensemble du pipeline

```
Microphone (ADC, 16kHz, int16 brut)
  -> buffer de 1024 échantillons (64ms), un par interruption DMA
  -> [mode EMA] gain RMS glissant (normalisation causale, waveform-domain, §4)
  -> fenêtrage de Hann + FFT 256 points -> magnitude
  -> masquage des bins < 150Hz (passe-haut, §5)
  -> [mode EMA/NONE] log10 (approx. fixe) ; [mode PCEN, déployé actuellement] compression PCEN
     par bin (§6), sur la magnitude - pas de log du tout, pas de gain waveform-domain en amont
  -> spectrogramme [7 frames x 129 bins] pour ce buffer
  -> quantification en int8
  -> 5 étages de convolution (CNN séparable en profondeur) -> [64 canaux, T'=1]
  -> moyennage sur l'axe fréquentiel (AdaptiveAvgPool2d((1,None)))
  -> un pas de LSTM (état caché/cellule persistant d'un buffer à l'autre)
  -> classifieur linéaire -> logit -> sigmoïde -> probabilité
```

Le point clé architectural : avec ce modèle, chaque buffer de 64ms produit **exactement un**
logit (les 3 étages à stride 2 réduisent le temps 7 -> 1). C'est ce qui permet un vrai
streaming : un pas de LSTM par buffer, état conservé, jamais réinitialisé pendant tout
l'enregistrement.

`STFT_NORMALIZATION_MODE` (`main.c`) choisit EMA/NONE/PCEN à la compilation - un seul mode
compilé à la fois, pas un choix runtime. **PCEN est le mode réellement déployé aujourd'hui**
(voir §6) ; EMA reste documenté (§4) car c'est le schéma historique, plus simple, et sert de
comparaison. Le piège de parité le plus grave rencontré sur ce projet (§14.5) vient précisément
d'un endroit où le code d'entraînement appliquait encore le gain EMA **avant** PCEN alors que
le firmware ne le fait jamais dans ce mode - lisez §6 et §14.5 ensemble avant de toucher à la
normalisation.

---

## 2. Le matériel et ses implications

- **Flash** : 256KB, accédée avec des "wait states" configurés par `flashWaitStateControl()`
  (`em_cmu.c`) - à 48MHz le CPU utilise WS2 (~2 cycles d'attente + 1 cycle d'accès = ~3 cycles
  par lecture), plus un mécanisme de préchargement qui aide sur du code/data séquentiel.
- **RAM interne** : 32KB, accès rapide (pas d'attente), mais très limitée - à peine quelques KB
  restent libres une fois le firmware normal chargé.
- **SRAM externe** : 256KB, mais accédée via EBI (External Bus Interface). `EBI_Init()`
  (`audiomoth.c`) configure `readSetupCycles=2 + readStrobeCycles=3 + readHoldCycles=1` = 6
  cycles par accès - **plus lent que la flash**, contrairement à l'intuition "RAM = plus
  rapide". Utile uniquement parce qu'elle offre du volume, pas de la vitesse : on y place les
  buffers d'activation trop gros pour la RAM interne (voir section 9), pas du code ou des
  données chaudes.
- **Conséquence pratique** : tout ce qui est lu/écrit souvent (poids du modèle, buffers de
  travail des kernels CMSIS-NN) doit rester en flash (pour les poids, en lecture seule) ou en
  RAM interne (pour les buffers modifiés). La SRAM externe est un espace de dernier recours
  pour des tampons volumineux, peu réutilisés dans la boucle chaude.

---

## 3. La STFT (CMSIS-DSP)

### 3.1 Paramètres

FFT 256 points, fenêtre de Hann symétrique (`0.5 - 0.5*cos(2*pi*i/(N-1))`), hop de 128
échantillons (50% de recouvrement) -> 7 frames x 129 bins (129 = 256/2 + 1, sortie d'une FFT
réelle) par buffer de 1024 échantillons (64ms).

### 3.2 `arm_rfft_fast_f32` et le piège de l'init générique

CMSIS-DSP fournit `arm_rfft_fast_f32()` pour une FFT réelle rapide. Avant de pouvoir l'appeler,
il faut initialiser une structure `arm_rfft_fast_instance_f32` avec les tables de calcul
nécessaires (facteurs de rotation "twiddle" et table de bit-reversal pour l'algorithme
Cooley-Tukey). La fonction générique pour ça est `arm_rfft_fast_init_f32(instance, fftLen)`.

**Le piège** : cette fonction est un simple `switch` sur `fftLen`, qui référence *toutes* les
tables possibles (16 à 4096 points), même si vous n'utilisez que 256. Comme `fftLen` est un
argument passé à l'exécution (même si dans votre code c'est toujours la même constante), le
compilateur ne peut pas prouver quelle branche du switch sera prise, et le linker
(`--gc-sections`) ne peut donc éliminer aucune des tables inutilisées. Résultat mesuré : ~75KB
de flash gaspillés en tables jamais utilisées.

**Solution** : CMSIS-DSP expose aussi des fonctions d'init spécifiques à chaque taille,
`arm_rfft_fast_init_256_f32()` ici. En l'appelant directement, seule la table des 256 points
est référencée, et tout le reste disparaît au link. Ajoutez un `#error` de garde si
`STFT_FFT_LENGTH != 256` pour ne jamais oublier de changer cet appel si la taille de FFT
change un jour.

**Leçon générale à retenir** : dans une bibliothèque comme CMSIS-DSP/CMSIS-NN, méfiez-vous de
toute fonction "générique" qui dispatche sur un paramètre runtime alors que dans votre usage
c'est une constante de compilation - vérifiez toujours avec `arm-none-eabi-nm --print-size
--size-sort --radix=d` sur le binaire final ce qui est réellement linké, ne présumez jamais.

### 3.3 Approximation log10 en virgule fixe

Le Cortex-M4F n'a pas d'unité matérielle pour les fonctions transcendantes (log, exp...) -
`log10f()` (libm) est ~21x plus cher qu'une approximation en virgule fixe, et
`arm_vlog_f32` (CMSIS-DSP) retombe sur un `logf()` scalaire (pas de MVE/Helium sur ce cœur),
~13x plus cher aussi.

Astuce classique IEEE-754 : pour un `float32` positif `x`, son motif binaire brut `i`
(réinterprété comme un entier) satisfait approximativement `log2(x) ~= i / 2^23 - 127`
(l'exposant IEEE-754 code déjà, de façon quasi-linéaire, `log2(x)`). Comme
`(i / 2^23) * 2^16 == i >> 7`, ceci donne `log2(x)` en virgule fixe Q16.16 avec un simple
décalage et une soustraction - aucune division/multiplication flottante. `log10(x)` s'obtient
ensuite par une seule multiplication en virgule fixe Q16.16 par `log10(2)`.

Le biais naïf de 127 est exact seulement aux puissances de deux ; la correction empirique
`126.95696433397204` recentre l'erreur autour de zéro, réduisant l'erreur max de 0.52dB à
0.26dB sur le résultat final `20*log10(x)`, pour le même coût.

Voir `stftFastLog2Q16_16()`/`stftFastLog10()` dans `main.c`, et son port bit-exact côté Python
d'entraînement, `scripts/fast_log10.py` / `student_model.py::_fast_log10_db()` - **crucial**
que l'entraînement voie exactement cette même approximation, pas `math.log10` exact, sinon
c'est un vrai écart entraînement/inférence.

### 3.4 STFT par buffer, pas en continu

Le firmware ne calcule jamais une STFT continue sur tout un enregistrement : chaque
interruption DMA traite un buffer de 1024 échantillons indépendamment, et recommence le
comptage des hops à 0 à chaque buffer (`source[frame*hop_length : frame*hop_length+n_fft]`
pour `frame` dans `0..6`, indices toujours relatifs au buffer courant). Aucune frame ne
s'étend jamais sur deux buffers.

Si votre code d'entraînement fait une STFT continue sur tout le clip (ex.
`torchaudio.transforms.Spectrogram` par défaut), il calcule **une frame de plus par frontière
de buffer** que ce que le firmware calculera jamais (celle qui chevauche la frontière) - un
écart entraînement/inférence discret mais réel. `student_model.py::LogSpectrogram.forward()`
reproduit exactement le découpage par buffer (voir le code, `buffer_samples=1024`).

---

## 4. Normalisation RMS glissante (causale)

**Cette section décrit `STFT_NORMALIZATION_EMA` - pas le mode actuellement déployé (`PCEN`,
§6).** Gardée en détail parce que le mécanisme (gain causal buffer par buffer) et ses pièges
(jitter de calibration non-seedé, notamment) restent pertinents ailleurs dans le pipeline, et
parce qu'EMA reste une comparaison utile. **Le gain décrit ici ne s'applique JAMAIS en mode
PCEN** - `computeBufferGain()` y retourne un `1.0f` fixe (voir §6) ; l'avoir appliqué quand même
côté entraînement, dans un chemin de code qui a oublié cette exclusion, est le bug de parité le
plus grave rencontré sur ce projet (§14.5) - lisez cette section avant de modifier quoi que ce
soit ici.

Le modèle a besoin d'un niveau de signal à peu près stable pour bien fonctionner. Faire ça sur
tout un clip (RMS du clip entier) n'est pas possible en temps réel (il faudrait connaître le
futur). La solution causale : une moyenne exponentielle (EMA) de la puissance moyenne
(mean-square) de chaque buffer, mise à jour une fois par buffer, transformée en gain linéaire
appliqué à ce buffer avant fenêtrage/FFT.

```c
// résumé de computeBufferGain() (main.c)
bufferMeanSq = mean(source[i]^2 pour i dans le buffer)   // échelle int16 brute
runningMeanSq = alpha * bufferMeanSq + (1-alpha) * runningMeanSq
gain = targetRmsRaw / sqrt(runningMeanSq + eps)
```

avec `alpha = 2/(N+1)`, `N` choisi pour que la constante de temps de l'EMA corresponde à la
durée d'un clip d'entraînement (3s ici, soit ~46.875 buffers de 64ms) - convention standard
pour "cette moyenne exponentielle se comporte comme une moyenne glissante sur N échantillons".
Le gain est replié directement dans la multiplication par la fenêtre de Hann
(`stftFrame[i] = source[i] * window[i] * gain`), jamais appliqué en modifiant le buffer DMA
brut.

Côté entraînement, `dataset.py::streaming_rms_normalize()` reproduit exactement la même
formule, buffer par buffer, sur le waveform brut (échelle int16, **pas** `[-1,1]`).

**Piège associé** : si votre code d'entraînement charge l'audio via une librairie qui
normalise automatiquement en `[-1,1]` (ex. `soundfile.read(..., dtype="float32")`), alors que
le firmware travaille sur l'int16 brut, vous avez un facteur d'échelle constant de
`20*log10(32768) ~= 90.3dB` entre les deux - chargez toujours en `dtype="int16"` et gardez
l'échelle brute de bout en bout si le firmware fait pareil.

**Autre piège, plus subtil** : si l'entraînement simule un état RMS initial aléatoire (jitter)
pour chaque clip - utile pour voir des régimes RMS variés en cours de clip - ce jitter doit
être tiré d'un générateur aléatoire **seedé par l'index du clip**, pas de l'état global du
RNG. Sinon, deux exécutions séparées du script de calibration/export (`export_int8.py`) tirent
des valeurs différentes, et les échelles de quantification calculées lors de la calibration ne
sont plus reproductibles d'un export à l'autre.

---

## 5. Filtre passe-haut (masquage des bins)

Les basses fréquences (<150Hz, essentiellement du bruit de vent/manipulation + offset DC)
sont mises à zéro (pas juste atténuées) *avant* le passage au log - un cut "brick-wall". Avec
`bin_width = sample_rate/fft_length = 62.5Hz`, `highpass_bin = ceil(150/62.5) = 3`, donc les
bins 0/1/2 sont masqués. Le masquage doit se faire sur la magnitude, pas sur le log en sortie
(`log(0+eps)` donne un plancher constant, différent de "attenuer après coup").

---

## 6. Normalisation PCEN (`STFT_NORMALIZATION_PCEN`, mode actuellement déployé)

Alternative à la normalisation EMA (§4) : au lieu d'un gain global appliqué au waveform avant
la FFT, PCEN (Per-Channel Energy Normalization, Wang et al. 2017) compresse **après** la FFT,
**par bin de fréquence**, avec des paramètres appris (pas fixes) :

```
M[t] = (1-ema_alpha)*M[t-1] + ema_alpha*E[t]      // moyenne mobile causale de la puissance, par bin
pcen = (E[t]/(M[t]+eps)^alpha + delta)^r - delta^r
```

`alpha`, `delta`, `r`, `ema_alpha` sont appris **par bin** (129 valeurs chacun) pendant
l'entraînement (`student_model.py::LearnablePCEN`), pas une seule constante partagée. `M`
(l'estimation causale de puissance) est un état **persistant par bin**, jamais réinitialisé en
cours d'enregistrement - l'équivalent PCEN de `stftRunningMeanSq` de la §4, mais un scalaire par
bin plutôt qu'un seul scalaire global.

**Aucun gain waveform-domain en amont** : `computeBufferGain()` retourne un `1.0f` fixe en mode
PCEN, exactement comme en mode `NONE` - toute l'adaptation se fait après la FFT, sur la
magnitude par bin, pas avant. C'est déjà ce que fait `student_model.py` côté entraînement
(`NORMALIZATION_MODE=pcen` saute `streaming_rms_normalize()` entièrement) - **sauf dans un
chemin de code précis qui a longtemps fait l'inverse, le bug de parité le plus grave rencontré
sur ce projet, voir §14.5**.

Les bins masqués (§5) n'ont besoin d'aucun cas particulier : PCEN d'une entrée toujours nulle
vaut toujours exactement `delta^r - delta^r == 0`, quels que soient les paramètres appris ou
l'état de `M` - la même propriété "sortie fixe, indépendante de la magnitude" que chaque autre
variante de `STFT_LOG_MODE` exploite déjà pour ces bins.

### Approximation numérique : `stftFastPow`, pas juste `stftFastLog10`

`alpha` et `r` sont des exposants fractionnaires **appris par bin**, pas un exposant fixe -
`stftFastLog10` (bit-trick IEEE-754, réutilisé pour le mode EMA/log) ne suffit plus tel quel.
`stftFastPow(x, a) = exp2(a * log2(x))` réutilise le même bit-trick (`stftFastLog2Q16_16()`,
même constante de biais) et ajoute son inverse algébrique (`stftFastExp2Q16_16()`) - même
qualité d'approximation déjà validée pour `stftFastLog10` (~0.26dB / ~3% pire cas), pas un
nouveau schéma numérique. Validé de bout en bout contre PCEN exact en float64 sur un
enregistrement réel : erreur absolue moyenne 0.052, p99 0.135, sur une plage de sortie PCEN
d'environ 0-14.

### Calibration INT8 : calibrer sur un percentile, pas le max littéral

PCEN produit une distribution de sortie à **queue très lourde** : sur un vrai jeu de
calibration, médiane ~22, p90 ~111, p99 ~318, mais le **max** atteint ~6700 - plus de 200x la
médiane, dans bien moins de 0.1% des frames (un "attack overshoot" : quand `M` (lent) n'a pas
encore rattrapé un son fort qui vient de démarrer, le ratio `E/M^alpha` explose transitoirement
avant que `M` ne remonte). Calibrer l'échelle INT8 sur ce max littéral (`export_int8.py`,
comportement par défaut, `--calibration-percentile 100`) gaspille presque toute la résolution
INT8 sur des valeurs qui n'arrivent presque jamais, écrasant la plage réellement informative
(médiane/p90) dans une poignée de niveaux sur 256.

Mesuré sur `student_windowed_lstm_lstm_16_32_64_h64_pcen.pt` : calibration sur le max -> AP
frame INT8 réel 87.1% contre 96.7% en float32 (delta -9.6%, stable de 40 à 100 clips de
calibration, donc pas un artefact d'échantillon). Calibration sur le p99.9
(`--calibration-percentile 99.9`) -> AP INT8 96.0% (delta -0.8%), précision à rappel≥95% 62.1%
-> 83.0%. Un gain massif pour un seul flag - voir `export_int8.py --long-clips
--calibration-percentile 99.9` (et son propre docstring `calibrate()`/`calibrate_long()` pour
l'implémentation : échantillonnage aléatoire **seedé** - `CALIBRATION_SAMPLE_SEED` - pas le RNG
global, pour que deux calibrations du même checkpoint restent reproductibles bit à bit, même
piège que celui déjà documenté en §4/§14.5).

`compare_device_output.py --long-clips` (mêmes flags qu'`export_int8.py`) rejoue une capture
réelle du device contre `int8_sim.py` pour un checkpoint PCEN/clips longs - corrige au passage
un bug préexistant (le rebuild du `QuantizedModel` n'appelait jamais `load_learned_pcen()`,
donc calibrait silencieusement contre l'init PCEN non entraînée). Résultat réel obtenu : bonne
corrélation device/simulateur (le vrai firmware suit fidèlement `int8_sim.py`), mais tous deux
dérivent progressivement du modèle float32 sur une séquence continue longue (~26s, 414 buffers
sans reset) - une propriété apparemment inhérente à l'état récurrent LSTM en précision réduite
sur une longue séquence, partagée par le device et la simulation, pas un bug firmware isolé.

---

## 7. Le modèle : TinyLSTMCNN

Architecture (`student_model.py::TinyLSTMCNN`, canaux 16/32/64, LSTM hidden=16) :

| Étage | Type | in_ch | out_ch | kernel | stride | pad | groups |
|---|---|---|---|---|---|---|---|
| stage0 | conv classique | 1 | 16 | 3x3 | 2 | 1 | 1 |
| stage1_dw | depthwise | 16 | 16 | 3x3 | 2 | 1 | 16 |
| stage1_pw | pointwise 1x1 | 16 | 32 | 1x1 | 1 | 0 | 1 |
| stage2_dw | depthwise | 32 | 32 | 3x3 | 2 | 1 | 32 |
| stage2_pw | pointwise 1x1 | 32 | 64 | 1x1 | 1 | 0 | 1 |

Chaque conv est suivie de BatchNorm (fusionné dans les poids avant quantification, voir §9) et
ReLU. Avec l'entrée `[1, 129, 7]` (fréquence x temps), les 3 étages à stride 2 réduisent le
temps `7 -> 4 -> 2 -> 1` : **un seul pas de temps en sortie par buffer**.

Ensuite : `AdaptiveAvgPool2d((1, None))` (moyenne sur l'axe fréquence uniquement, garde le
temps) -> LSTM à une couche (`nn.LSTM(64, 16, batch_first=True)`) -> classifieur linéaire
(`nn.Linear(16, 1)`).

---

## 8. Quantification INT8 : les notions de base

Toute cette section est indépendante de CMSIS-NN - ce sont les conventions génériques de
quantification "affine" utilisées par TFLite et reprises telles quelles par CMSIS-NN.

### 7.1 Scale et zero-point

Une valeur réelle `r` est représentée par un entier quantifié `q` via `r = scale * (q -
zero_point)`. Ici, **zero_point est toujours 0** (quantification "symétrique") : `r = scale *
q`. C'est un choix de simplicité (pas de terme de correction `zero_point * somme_des_poids` à
gérer dans les convolutions), au prix d'un poil de précision (le 0 n'est pas forcément le
centre optimal de la plage).

### 7.2 Scale par canal (poids conv) vs par tenseur (poids LSTM)

- **Poids de convolution** : quantifiés **par canal de sortie** (`per_channel_scale`) - chaque
  filtre de sortie a sa propre échelle, ce qui capture mieux la variance de poids entre
  filtres qu'une échelle unique pour toute la couche. C'est pour ça que
  `cmsis_nn_per_channel_quant_params` a un tableau `multiplier[]`/`shift[]` (un par canal), pas
  un scalaire.
- **Poids LSTM** : quantifiés **par tenseur** (une seule échelle pour toute la matrice
  input-to-hidden, une autre pour toute la matrice hidden-to-hidden) - `cmsis_nn_lstm_gate` n'a
  qu'un `input_multiplier`/`input_shift` scalaire par porte, pas un tableau. Ne réutilisez donc
  pas la même fonction de quantification par canal pour les poids LSTM.

### 7.3 Le "multiplier/shift" : comment CMSIS-NN représente un facteur réel

Après une multiplication matricielle en int8, l'accumulateur est en int32, à une échelle
`input_scale * weight_scale` qu'il faut ramener à l'échelle de sortie voulue
(`output_scale`), en multipliant par le facteur réel `input_scale * weight_scale /
output_scale`. Ce facteur est un flottant quelconque, mais CMSIS-NN (comme TFLite) n'utilise
**aucune multiplication flottante à l'exécution**, uniquement de l'arithmétique entière.

`quantize_multiplier()` (convention TFLite `QuantizeMultiplier`, portée dans `int8_sim.py`)
décompose ce facteur réel positif en :
- un entier `multiplier` (mantisse Q31, dans `[0.5, 1.0)` représentée en entier 32 bits via
  `frexp` + arrondi),
- un `shift` (exposant),

tels que `facteur_reel = multiplier * 2^-31 * 2^shift`. À l'exécution, appliquer ce facteur à
un accumulateur `acc` (int32) revient à :

```
acc_scaled = (acc * multiplier) >> 31   // "doubling high multiply" - voir arm_nn_requantize
result     = round_shift(acc_scaled, shift)  // décalage à droite avec arrondi au plus proche
```

C'est exactement ce que fait `arm_nn_requantize()` (`CMSIS-NN/Include/arm_nnsupportfunctions.h`) :
1. `arm_nn_doubling_high_mult_no_sat(val, multiplier)` = `((1<<30) + val*multiplier) >> 31`
   (le "doubling" vient du fait que multiplier deux Q31 en gardant la moitié haute des 62 bits
   revient à retrouver la précision perdue par la représentation `[0.5,1)` - c'est la manière
   standard ARM de faire une multiplication flottante avec de l'entier pur, arrondie au plus
   proche via le `+ (1<<30)`).
2. `arm_nn_divide_by_power_of_two(high_mult, shift)` = décalage à droite par `shift`, avec
   arrondi au plus proche s'éloignant de zéro pour les cas à mi-chemin (pas une simple
   troncature).

Retenez juste : **`arm_nn_requantize(acc, multiplier, shift)` est LA fonction à utiliser
partout où un accumulateur int32 doit être ramené à une échelle de sortie int8/int16** - ne
réinventez pas une version approximative (division flottante, etc.), sinon vos résultats
divergeront subtilement du "vrai" calcul quantifié que le firmware final exécute (c'est
exactement pour ça que `int8_sim.py` réimplémente `arm_nn_requantize` ligne à ligne plutôt que
d'utiliser une division flottante "équivalente").

### 7.4 Quantification du biais

Le biais d'une conv/linéaire est quantifié dans le domaine de l'accumulateur **avant**
requantization : `bias_q = round(bias_float / (input_scale * weight_scale))`, ajouté à
l'accumulateur int32 avant l'appel à `arm_nn_requantize`. C'est la convention standard
TFLite/CMSIS-NN, reprise partout dans ce projet (`export_int8.py::quantize()` pour les conv,
voir §11 pour le cas particulier - et piégeux - du LSTM).

---

## 9. CMSIS-NN : conventions et fonctions de convolution

CMSIS-NN est la bibliothèque ARM de kernels de réseaux de neurones quantifiés (int8, et int16
pour certaines variantes), optimisée pour les cœurs Cortex-M (SIMD 16 bits type SMLAD sur
Cortex-M4/M4F, ou MVE/Helium sur Cortex-M55+). Elle fournit des fonctions comme
`arm_convolve_s8`, `arm_depthwise_conv_s8`, `arm_fully_connected_s8`, `arm_nn_lstm_step_s8`,
etc., chacune prenant des poids/biais/paramètres de quantification déjà préparés (pas de
logique de quantification *dans* CMSIS-NN elle-même - c'est à vous de la faire, comme dans
`export_int8.py`).

### 8.1 Structures communes

- `cmsis_nn_context` : `{ void *buf; int32_t size; }` - un buffer scratch générique que
  certains kernels demandent (voir `arm_convolve_s8`/`arm_depthwise_conv_s8_opt` ci-dessous).
- `cmsis_nn_dims` : `{ n, h, w, c }` - **l'ordre des champs indique le layout mémoire attendu :
  channel-minor (NHWC)**, c'est-à-dire que pour une position spatiale (y, x) donnée, tous les
  canaux sont contigus en mémoire : `index = (y*w + x)*channels + c`. C'est l'opposé du layout
  natif de PyTorch (`NCHW`, channel-major : `index = (c*h + y)*w + x`). **Toute donnée passée
  à/reçue de CMSIS-NN doit être en NHWC** - un piège classique si vous partez d'un tenseur
  PyTorch sans le repermuter.
- `cmsis_nn_conv_params` / `cmsis_nn_dw_conv_params` : stride/padding/dilation/offsets
  (input_offset, output_offset - toujours 0 ici) + bornes d'activation
  (`activation.min`/`activation.max`, ex. `[0, 127]` pour fusionner un ReLU directement dans la
  requantization).
- `cmsis_nn_per_channel_quant_params` : `{ int32_t *multiplier; int32_t *shift; }` - un
  multiplicateur/shift par canal de sortie (voir §8.3).

### 8.2 `arm_convolve_s8` : convolution générale, et le piège de l'im2col

`arm_convolve_s8` implémente une convolution 2D int8 générique via la technique **im2col** :
pour chaque position de sortie, elle "déroule" (gather) la fenêtre d'entrée correspondante en
une colonne contiguë en mémoire (un buffer scratch), puis effectue le calcul comme une
multiplication matricielle dense entre ce buffer et les poids (déjà aplatis en
`[out_ch, kh*kw*in_ch]`). C'est une technique classique pour réutiliser un noyau de matmul
générique et optimisé plutôt que d'écrire une boucle de convolution spécialisée.

**Le piège rencontré ici** : ce projet a un étage (`stage0`) avec `in_ch=1`. Pour chaque
position de sortie, l'étape d'im2col doit gérer le padding : recopier les valeurs d'entrée
valides, et écrire des zéros là où la fenêtre déborde (padding). Avec `in_ch=1`, chaque
opération de recopie/mise-à-zéro ne déplace qu'un seul octet (`arm_memcpy_s8`/
`arm_memset_s8`, des wrappers), et sur Cortex-M4 (pas de MVE), ces wrappers retombent sur des
appels génériques `memcpy()`/`memset()` de la libc - un vrai appel de fonction pour déplacer un
octet, répété `(H_out * W_out) * (kh*kw)` fois. Le calcul arithmétique lui-même est
négligeable (37440 MACs) ; c'est l'overhead d'appel de fonction qui domine. Mesuré : ~41% du
temps total d'inférence sur ce seul étage.

**Solution retenue ici** : écrire à la main une boucle de convolution directe pour ce cas
précis (`in_ch=1`), lisant les 9 valeurs du noyau directement dans le tenseur d'entrée, sans
buffer im2col du tout. Ceci n'est possible/correct **que parce que `input_offset` (le
zero-point d'entrée) est 0** dans tout ce modèle - une position de padding contribue alors
`0 * poids = 0` à la somme, qu'on peut simplement *sauter* au lieu de la mettre à zéro
explicitement (voir §12.3 pour le code). Ne faites jamais ça si votre modèle a un
`input_offset` non nul (asymétrique) : il faudrait alors ajouter `input_offset * poids` pour
chaque tap manquant, ce qui annule tout le bénéfice.

Pour les 2 étages "pointwise" (1x1, stride 1, pad 0), im2col ne fait presque rien (une fenêtre
1x1 = juste le pixel lui-même, pas de gather réel) donc `arm_convolve_s8` reste la meilleure
option mesurée ici - voir §12 pour la tentative avortée de le remplacer.

### 8.3 `arm_depthwise_conv_s8` vs `arm_depthwise_conv_s8_opt`

Une convolution "depthwise" applique un filtre séparé à chaque canal d'entrée (pas de mélange
entre canaux) - `groups == in_channels`. CMSIS-NN a deux implémentations :

- **`arm_depthwise_conv_s8`** : implémentation générale, supportant un "multiplicateur de
  canal" (`ch_mult`, càd `out_channels = in_channels * ch_mult` pour un ch_mult>1). Son chemin
  SIMD accéléré (`depthwise_conv_s8_mult_4`) n'est activé **que si `ch_mult % 4 == 0`**
  (confirmé directement dans `arm_depthwise_conv_s8.c`, ligne ~291 :
  `if (dw_conv_params->ch_mult % 4 == 0 && ...)`) ; sinon, elle retombe sur
  `depthwise_conv_s8_generic`, un chemin **entièrement scalaire, non-SIMD**.
- **`arm_depthwise_conv_s8_opt`** : variante dédiée au cas `ch_mult == 1` (canal préservé,
  `in_channels == out_channels`, le doc du fichier dit littéralement *"channel multiplier of
  1"*), avec un chemin DSP/SMLAD accéléré. Elle exige `output_ch % 4 == 0` pour éviter sa
  propre boucle scalaire résiduelle (vrai ici : 16 et 32 sont tous deux multiples de 4), et
  a besoin d'un petit buffer scratch (`Cin * kH * kW * sizeof(int16_t)`, ex. 576 octets pour un
  étage 32 canaux / noyau 3x3).

Ce modèle a des étages depthwise **channel-preserving** (`ch_mult=1`), c'est-à-dire
exactement le cas que `arm_depthwise_conv_s8`'s condition SIMD rate systématiquement (1%4 !=
0). Résultat : utiliser `arm_depthwise_conv_s8` "par défaut" fait tomber ces deux étages sur le
chemin scalaire le plus lent mesuré du pipeline (~1008 ns/MAC). Passer à
`arm_depthwise_conv_s8_opt` (même layout de poids, signature légèrement différente, aucun
ré-export nécessaire) a donné -25% et -21% sur ces deux étages.

**Règle pratique à retenir** : pour une conv depthwise "classique" (canal préservé,
`ch_mult=1`), utilisez systématiquement `arm_depthwise_conv_s8_opt`, jamais la version
générique - et vérifiez `output_ch % 4 == 0`, sinon vous retombez sur sa boucle résiduelle
scalaire (toujours plus rapide que la version generic, mais moins bon que le cas nominal).

### 8.4 Pas de kernel pour "pooling fréquence seule"

`arm_avgpool_s8` moyenne sur **les deux** axes spatiaux (H et W) - il n'existe aucune fonction
CMSIS-NN pour "moyenner H, garder W" (ce dont ce modèle a besoin :
`AdaptiveAvgPool2d((1, None))`). Il faut l'écrire à la main : accumulation int32 canal par
canal sur l'axe H, division avec arrondi (`(sum >= 0 ? sum + h/2 : sum - h/2) / h`), pas de
multiplier/shift nécessaire (le pooling ne change pas l'échelle - moyenner ne fait que
diviser, ce que fait déjà directement la division entière).

**Piège du layout (NCHW vs NHWC), rencontré ici en vrai** : la sortie des kernels de
convolution CMSIS-NN (`arm_convolve_s8`, `arm_depthwise_conv_s8_opt`) est en NHWC
(channel-minor : `index = (y*w+x)*channels + c`), *pas* en NCHW (channel-major, le layout
natif PyTorch). Une première version de la fonction de pooling manuelle l'indexait en NCHW
(`(c*h+y)*w+x`) - silencieusement fausse dès que `channels > 1` (les deux formules produisent
des indices valides, donc ni une erreur de compilation ni un plantage ne l'auraient révélé) :
confirmé par un test numérique direct contre `int8_sim.py` (erreur max 29 sur une échelle
0-127). **Vérifiez toujours le layout réel en lisant le code source du kernel qui produit le
tenseur que vous consommez**, ne vous fiez pas à une intuition "comme en PyTorch".

### 8.5 Classifieur final : `arm_fully_connected_s8` ou dot product manuel

Pour une seule sortie scalaire (`nn.Linear(hidden_size, 1)`), un dot product manuel int32
(`sum(weight[i] * hidden[i]) + bias`) suivi d'une conversion directe en flottant via un
facteur d'échelle (`logit_scale = hidden_scale * weight_scale`) est plus simple qu'appeler
`arm_fully_connected_s8` - et évite une requantification int8 inutile pour une sortie qui va
de toute façon être reconvertie en flottant pour la sigmoïde. Voir `runClassifier()` dans
`model_inference.c`.

---

## 10. Organisation mémoire des tenseurs d'activation

Le plus gros tenseur intermédiaire (sortie de `stage0`, `16 canaux * 65 * 4 = 4160 octets`) ne
rentre pas dans la RAM interne déjà largement occupée par le reste du firmware (~4-5KB
restants). Solution : deux buffers "ping-pong" (`activationBufferA`/`B`) placés en SRAM externe
(lente, mais avec largement la place), chaque étage écrivant dans l'un en lisant l'autre, en
alternance. Le tenseur d'entrée quantifié et le petit vecteur pooled (entrée du LSTM, 64
valeurs) restent en RAM interne (rapide, petits).

---

## 11. Les couches LSTM CMSIS-NN : LE point sensible

C'est ici qu'a eu lieu le bug le plus sérieux de toute l'investigation (biais silencieusement
faux sur chaque pas de LSTM, chaque buffer) - cette section mérite d'être lue attentivement
avant toute réimplémentation.

### 10.1 Vue d'ensemble : structures et convention de format

Un LSTM a 4 "portes" (forget, input, cell, output), chacune calculée comme :
`porte = activation(x @ W_input.T + h_prev @ W_hidden.T + bias)`, où `activation` est sigmoïde
(forget/input/output) ou tanh (cell). CMSIS-NN représente ça avec :

- `cmsis_nn_lstm_gate` (une instance par porte) : poids input/hidden (int8), multiplicateurs/
  shifts (scalaires, **quantification par tenseur**, voir §8.2), type d'activation, et - point
  crucial - `bias` **et** `input_effective_bias`/`hidden_effective_bias` (voir §11.3).
- `cmsis_nn_lstm_params` : dimensions (`input_size`, `hidden_size`, `batch_size`,
  `time_steps`), les 4 `cmsis_nn_lstm_gate`, plus les paramètres propres à la mise à jour de
  l'état cellule (`forget_to_cell_multiplier/shift`, `input_to_cell_multiplier/shift`,
  `cell_clip`, `cell_scale_power`) et à la sortie (`output_multiplier/shift`, `output_offset`).
- `cmsis_nn_lstm_context` : 3 buffers scratch (`temp1`, `temp2`, `cell_state`), chacun de
  `hidden_size * sizeof(int16_t)` - minuscule (32 octets pour hidden_size=16).

**Conventions de format fixes** (pas calibrées, imposées par le kernel) :
- La **pré-activation** de chaque porte (avant sigmoïde/tanh) est en **Q3.12** (échelle
  `2^-12`, valeur réelle = `int16_value / 4096`).
- La **post-activation** (sortie de sigmoïde/tanh) est en **Q0.15** (échelle `2^-15`).
- Seule l'**échelle de l'état cellule** est calibrée, exprimée comme une puissance de deux :
  `cell_scale = 2^cell_scale_power` (typiquement négatif, ex. -11 ou -12, puisque l'état
  cellule prend en général des valeurs de quelques unités et qu'int16 offre bien plus de
  marge que ça).

### 10.2 `arm_nn_lstm_step_s8` : le déroulé exact d'un pas

Code réel (`CMSIS-NN/Source/NNSupportFunctions/arm_nn_lstm_step_s8.c`), résumé :

```c
forget_gate = calculate_gate(data_in, hidden_in, forget_gate_params)   // temp1
cell_state = elementwise_mul(forget_gate, cell_state, forget_to_cell_mult/shift)  // 1er terme, en place

input_gate = calculate_gate(data_in, hidden_in, input_gate_params)     // temp1 (réutilisé !)
cell_gate  = calculate_gate(data_in, hidden_in, cell_gate_params)      // temp2

cell_state += elementwise_mul(input_gate, cell_gate, input_to_cell_mult/shift, clip=[-cell_clip,cell_clip])

output_gate = calculate_gate(data_in, hidden_in, output_gate_params)   // temp1 (réutilisé !)

hidden_temp = tanh_lut(cell_state, shift=cell_scale_power+12)   // vers Q3.12 puis LUT tanh -> Q0.15, dans temp2
hidden_out  = elementwise_mul_s16_s8(output_gate, hidden_temp, output_multiplier/shift)  // -> int8
```

**Détail important si vous réimplémentez ça vous-même** : le vrai kernel réutilise
agressivement `temp1` pour forget/input/output (jamais utilisés simultanément) et `temp2` pour
cell_gate puis pour `hidden_temp` - une optimisation mémoire à connaître si vous écrivez votre
propre version (ou si vous voulez comprendre pourquoi seulement 2 buffers scratch suffisent
pour 4 portes).

Le calcul de `hidden_temp` fait `cell_scale_power + 12` : c'est la conversion de l'état cellule
(échelle `2^cell_scale_power`) vers Q3.12 (échelle `2^-12`) avant de passer dans la table tanh,
qui attend toujours du Q3.12 en entrée - un décalage de `cell_scale_power - (-12) =
cell_scale_power + 12` bits.

### 10.3 LE PIÈGE : `bias` n'est jamais lu, seul `*_effective_bias` compte

`arm_nn_lstm_calculate_gate_s8_s16()` (appelée pour chaque porte) :

```c
arm_cmsis_nn_status arm_nn_lstm_calculate_gate_s8_s16(...) {
    memset(output, 0, ...);
    arm_nn_vec_mat_mul_result_acc_s8_s16(data_in, gate->input_weights,
                                          gate->input_effective_bias,   // <-- ICI
                                          output, gate->input_multiplier, gate->input_shift, ...);
    if (hidden_in) {
        arm_nn_vec_mat_mul_result_acc_s8_s16(hidden_in, gate->hidden_weights,
                                              gate->hidden_effective_bias,  // <-- ET ICI
                                              output, gate->hidden_multiplier, gate->hidden_shift, ...);
    }
    arm_nn_activation_s16(output, output, ..., gate->activation_type);
}
```

Le champ `gate->bias` **n'apparaît nulle part** dans cette fonction (ni dans aucune autre
fonction du chemin d'exécution de `arm_nn_lstm_step_s8`). Seuls `input_effective_bias` et
`hidden_effective_bias` sont utilisés, chacun ajouté comme biais de départ de l'accumulateur
dans `arm_nn_vec_mat_mul_result_acc_s8_s16()` (`acc_0 = *effective_bias_ptr++` en tout début de
boucle, avant la multiplication matrice-vecteur), **avant** la requantization
`arm_nn_requantize(acc, dst_multiplier, dst_shift)` propre à cette passe (input ou hidden).

**Pourquoi deux biais effectifs (input et hidden) et pas un seul ?** Parce que le biais réel
ne doit être ajouté **qu'une seule fois** au total (pas une fois par passe), mais les deux
passes (input-to-hidden, hidden-to-hidden) sont requantifiées séparément avec des
multiplier/shift différents (les deux matrices n'ont pas la même échelle de poids). La
convention retenue : tout le biais est mis dans `input_effective_bias` (ajouté pendant la passe
input, dans le domaine pré-requantization de **cette** passe), et `hidden_effective_bias` est
**toujours un tableau de zéros** (rien à ajouter pendant la passe hidden).

**Le bug concret rencontré ici** : le code de câblage (`runLSTMStep()` dans
`model_inference.c`) ne renseignait que `gate->bias` (le champ jamais lu) et laissait
`input_effective_bias`/`hidden_effective_bias` à `NULL`. Sur ce cœur (Cortex-M), l'adresse 0
correspond au début de la table de vecteurs d'interruption en flash (pas une zone non-mappée
qui déclencherait un fault) - donc `*effective_bias_ptr++` sur un pointeur NULL ne plante pas,
il lit silencieusement des octets de la table de vecteurs comme si c'était le biais. Résultat :
chaque porte, à chaque pas de LSTM, ajoutait une valeur de biais complètement fausse (mais
plausible, dans la bonne plage numérique) - un bug "silencieux" qui a nécessité un test de
vérification sur device dédié pour être détecté (voir §13).

**Comment calculer `input_effective_bias` correctement** (convention TFLite/CMSIS-NN standard,
confirmée en lisant le générateur de tests officiel de CMSIS-NN,
`Tests/UnitTest/lstm_settings.py::calc_effective_bias()`) :

```
input_effective_bias[j] = round(bias_float[j] / (input_scale * input_weight_scale))
                           (+ zero_point_correction, toujours 0 ici car input_offset=0)
hidden_effective_bias[j] = 0   (pour toutes les 4 portes)
```

C'est **exactement la même valeur** de biais que celle utilisée par `bias_q3_12`
(utilisé par le simulateur Python `int8_sim.py`, qui l'ajoute lui-même après les deux
requantizations, dans le domaine Q3.12 fixe) - juste exprimée dans un domaine différent (celui
de l'accumulateur brut, avant multiplication/shift, plutôt qu'après). Les deux formulations
sont mathématiquement équivalentes ; ce qui compte, c'est de fournir au vrai kernel la version
qu'il attend (`input_effective_bias`), pas celle utilisée par votre simulateur de référence.

**Règle générale à en tirer** : ne faites jamais confiance au nom d'un champ de structure
("bias" semble évident) sans vérifier, dans le code source du kernel qui la consomme, quels
champs sont **réellement lus**. Un champ qui existe dans une structure documentée peut très
bien n'être là que pour un usage différent (ici : uniquement utile à un simulateur de test
CMSIS-NN interne, pas au kernel réel).

### 10.4 Activation : table de correspondance (LUT), pas un vrai calcul flottant

`arm_nn_activation_s16()` calcule sigmoïde/tanh via une table de correspondance
(`sigmoid_table_uint16`) avec interpolation linéaire entre entrées adjacentes - jamais un vrai
calcul flottant de `exp()`/`tanh()`. `tanh(x) = 2*sigmoid(2*x) - 1` est utilisé pour dériver
tanh à partir de la même table que sigmoïde (un seul LUT pour les deux activations, avec un
décalage d'échelle différent selon le type - `abs_input_shift = 9` pour sigmoïde, `8` pour
tanh).

**Conséquence pour la validation** : un simulateur Python "propre" qui calcule sigmoïde/tanh
avec de vrais flottants (comme le fait `int8_sim.py`, pour rester simple et lisible) ne sera
**jamais bit-exact** avec le vrai kernel - un écart résiduel de quelques unités par élément est
attendu et normal (mesuré : ~7/32 éléments différents d'exactement 1 unité, après correction du
bug du §11.3, contre 22-26/32 avant). Ne cherchez pas à éliminer ce bruit résiduel si vous
utilisez un simulateur simplifié comme oracle - c'est la limite connue de l'approche, pas un
bug restant.

### 10.5 Dérivation des multiplicateurs `forget_to_cell`/`input_to_cell`

La mise à jour de l'état cellule combine deux produits à des échelles différentes qu'il faut
ramener à l'échelle calibrée `2^cell_scale_power` :
- `forget_gate` (Q0.15, échelle `2^-15`) `* cell_prev` (échelle `2^cell_scale_power`) -> produit
  à l'échelle `2^(-15+cell_scale_power)`, à requantifier vers `2^cell_scale_power` : facteur
  réel = `2^-15` (le `cell_scale_power` s'annule).
- `input_gate` (Q0.15) `* cell_gate` (Q0.15, sortie de tanh) -> produit à l'échelle `2^-30`, à
  requantifier vers `2^cell_scale_power` : facteur réel = `2^(-30-cell_scale_power)`.

Ces deux facteurs réels passent chacun par `quantize_multiplier()` (§8.3) pour donner
`forget_to_cell_multiplier/shift` et `input_to_cell_multiplier/shift`. Voir
`int8_sim.py::cell_multipliers()` pour l'implémentation canonique (partagée entre le
simulateur et l'export C, pour être sûr que les deux ne divergent jamais).

---

## 12. Optimisations de performance mesurées

Toutes mesurées sur le vrai matériel (jamais en simulation) - leçon répétée plusieurs fois
dans cette investigation : **une modification qui "devrait" être plus rapide en lisant le code
source ne l'est pas forcément une fois linkée dans le binaire réel** (effets de layout flash,
prefetch, etc. invisibles au niveau du code).

| Changement | Résultat |
|---|---|
| `arm_depthwise_conv_s8` -> `arm_depthwise_conv_s8_opt` | -25%/-21% sur les 2 étages depthwise |
| `runStage0Conv` écrit à la main (bypass im2col, `in_ch=1`) | -45% sur stage0, -18.6% sur l'inférence totale |
| `arm_convolve_wrapper_s8` pour les étages pointwise | **Rejeté** : +41%/+75% (plus lent) |
| Kernel pointwise écrit à la main (SIMD, sans im2col) | **Rejeté** : correct (0 écart), mais net -1.2ms sur le pipeline entier (déplacement du layout `.text` a ralenti `stage0`, non touché) |

Détail de la 3e ligne : `arm_convolve_wrapper_s8` route en théorie les convs 1x1/stride-1
vers `arm_convolve_1x1_s8_fast` (pas d'im2col du tout), ce qui semblait strictement meilleur en
lisant le code - mesuré pourtant nettement plus lent sur ce cœur/compilateur/forme précis.

Détail de la 4e ligne : même en prouvant la correction bit-exacte (0 écart mesuré sur
device), le résultat net sur le pipeline complet était pire, pas meilleur - ajouter/retirer des
fonctions change assez la disposition du code en flash pour affecter le comportement du
préchargement sur du code *non lié* à la modification. **Conclusion pratique : mesurez
toujours le pipeline entier après chaque changement, pas seulement l'étage modifié.**

---

## 13. Méthodologie de validation (comment vérifier chaque étape sans se tromper)

1. **`int8_sim.py`** : un simulateur Python pur (numpy), qui réimplémente ligne à ligne la
   même arithmétique que les vrais kernels CMSIS-NN (`arm_nn_requantize` porté exactement, pas
   approximé) - sert d'oracle de référence pour valider l'export avant même d'écrire du C.
2. **Test natif x86** avant tout flash : pour chaque remplacement de kernel écrit à la main
   (ex. `runStage0Conv`), compiler une version portable du code (pas les intrinsèques ARM-only)
   et la comparer bit-à-bit contre `arm_convolve_s8`/`int8_sim.py` sur des données aléatoires
   avant de jamais toucher au matériel réel.
3. **Vérification sur device** : pour les fonctions dépendant d'instructions ARM spécifiques
   (SIMD/DSP) qui ne compilent pas sur x86, ajouter temporairement un test embarqué qui exécute
   à la fois l'ancien et le nouveau chemin sur chaque buffer réel, compte les octets différents,
   et les expose via des colonnes CSV/binaires supplémentaires - à retirer une fois la
   correction confirmée.
4. **`compare_device_output.py`** : reconstruit le même `QuantizedModel` que celui exporté
   (mêmes checkpoint/calibration), rejoue les spectrogrammes réellement capturés par le
   firmware à travers `int8_sim.py`, en conservant l'état LSTM entre buffers (comme le fait
   `ModelInference_run()`), et compare logit par logit au résultat réel du device - c'est ce
   test qui a permis de détecter/diagnostiquer le bug du §11.3 (corrélation device/PC 0.32 avant
   correction, 0.9999 après).

---

## 14. Pièges d'entraînement (parité entraînement/déploiement)

Tout ce qui précède concerne le firmware et l'export - cette section couvre des bugs trouvés
**côté entraînement** (`test-audiomoth-cnn`, pas ce dépôt), découverts après coup en testant le
modèle déployé sur de vrais enregistrements de terrain. Le principe général qui les relie : la
fonction d'entraînement doit calculer **exactement** ce que le firmware calcule en déploiement
réel, pas une version "proche" ou "plus naturelle à coder" - deux fois de suite, une version
plus simple/continue à coder a silencieusement divergé d'une contrainte réelle du déploiement
streaming.

### 14.1 Le CNN doit être relancé par buffer à l'entraînement aussi

Le bug le plus significatif de cette investigation **avant la découverte du bug AGC/PCEN
(§14.5, plus grave encore)**, et plus grave que n'importe quel bug firmware du §16 (bugs)
ci-dessous : `TinyLSTMCNN.frame_logits()` (`student_model.py`) faisait
tourner les 5 étages de convolution **en continu** sur tout le spectrogramme d'un clip
d'entraînement (un seul appel `self.features(spectrogram.unsqueeze(1))` sur la séquence
entière) - donnant à chaque couche de conv un vrai champ réceptif temporel de plusieurs
buffers, via l'empilement de 3 convolutions à stride 2. Le firmware réel
(`ModelInference_run()`, §7-10 ci-dessus) relance ce même empilement **à zéro à chaque buffer de
64ms**, ne voyant jamais que les 7 frames STFT de ce buffer - aucun champ réceptif inter-buffer
pour les couches de conv, seul le LSTM porte un état entre buffers.

Ce n'est **pas** du bruit de quantification - un vrai écart structurel de calcul, confirmé sur
un extrait connu positif du jeu d'évaluation : confiance 0.6-0.95 sous l'évaluation continue,
~0.13-0.18 sous une réplique du calcul réel relancé par buffer, sur le **même** modèle flottant
non quantifié.

Corrigé en faisant en sorte que `frame_logits()` relance lui-même le CNN par buffer
(`_per_buffer_pooled_features()`, `student_model.py`) : reshape/permute le spectrogramme `[B,
F, T]` pour replier l'axe des buffers dans l'axe batch (`[B*num_buffers, 1, F,
frames_per_buffer]`), un seul appel batché à `self.features` (équivalent à `num_buffers` appels
séparés, mais bien plus rapide), puis reshape retour vers `[B, num_buffers, C]` avant de nourrir
le `temporal_context` (LSTM/Conv1d) **une seule fois, en continu**, sur toute la séquence - c'est
justement ce qui donne la mémoire inter-buffer, comme l'état LSTM persistant du firmware.

Piège associé : `export_int8.py` a **deux réimplémentations indépendantes** du même calcul
(`float_forward_with_traces()`, en `F.conv2d` brut sur poids BN-fusionnés ; `int8_forward()`, en
`sim.conv2d_s8`) qui ne passent PAS par `frame_logits()` - elles avaient exactement le même bug,
séparément, et ont eu besoin du même correctif séparément. Rien ne garantissait que ces
implémentations concordaient entre elles ; un test de cohérence directe
(`model.frame_logits(spec)` vs `float_forward_with_traces(...)["logits"]`) a été ajouté après
coup pour fermer ce risque.

**Leçon générale** : dès qu'un déploiement réel a une contrainte de calcul par morceaux
(streaming, buffer par buffer, fenêtre glissante...), **toute** réimplémentation du forward
pass - entraînement, évaluation, ré-export - doit respecter cette même contrainte exactement,
même si une version continue "look correcte" et s'entraîne sans erreur. Une divergence de ce
type ne casse rien syntaxiquement, elle dégrade juste silencieusement la qualité réelle du
modèle déployé.

### 14.2 La constante de temps de l'EMA ne doit jamais dépendre de la durée du clip d'entraînement

Bug plus subtil, découvert en corrigeant un autre problème (voir 13.3) : `dataset.py` calculait
`RMS_EMA_BUFFERS = CLIP_SECONDS / (BUFFER_SAMPLES / SAMPLE_RATE)` - dérivant la constante de
temps de la moyenne glissante RMS de la **durée du clip d'entraînement**. Mais côté firmware,
`STFT_RMS_EMA_BUFFERS = 46.875f` (`main.c`) est une constante **fixe**, ~3 secondes réelles,
sans aucune notion de "durée de clip" (un enregistrement réel n'en a pas). Les deux ne
coïncidaient que parce que `CLIP_SECONDS` valait justement 3.0 au moment où la formule a été
écrite - changer `CLIP_SECONDS` (par exemple pour donner au modèle de vrais clips plus longs,
voir 13.3) aurait silencieusement étiré aussi la constante de temps de l'EMA côté entraînement,
désynchronisant le comportement appris du comportement réel du firmware, sans qu'aucune erreur
ne le signale.

**Leçon générale** : la constante de temps d'une statistique causale/glissante est une propriété
du système déployé en temps réel (secondes réelles), jamais une fonction de la manière dont on a
découpé les données d'entraînement. Toujours vérifier qu'un tel paramètre reste **fixé, pas
dérivé**, si on touche à la façon dont les clips d'entraînement sont construits.

### 14.3 Diversité de volume dans les données d'entraînement

Pourquoi ce bug d'EMA a été trouvé : le modèle échouait sur un vrai enregistrement de terrain de
82s (voix humaine) qu'il reconnaissait bien avant ces changements. Diagnostic : des fenêtres de
3s extraites de ce même fichier, réamorcées avec leur propre RMS réel, obtiennent 0.70-0.77 de
confiance ; les mêmes instants évalués en continu sur le fichier entier (amorce EMA fixe,
dérivant pendant 82s) n'obtiennent que 0.10-0.27. Cause racine : les clips d'entraînement
(`CLIP_SECONDS=3.0` à l'époque) ne duraient qu'~1 constante de temps de l'EMA, et étaient
toujours amorcés proches du bon niveau (RMS réel du clip ± un facteur aléatoire) - le modèle
n'avait jamais vu de vraie **convergence en cours de clip** (estimation qui part fausse et
dérive vers le bon niveau en cours de route), alors que c'est exactement ce qui se passe sur un
vrai enregistrement continu de plusieurs minutes.

Corrigé en réentraînant sur des clips bien plus longs (`CLIP_SECONDS=12.0`, ~4 constantes de
temps de l'EMA - assez pour une vraie convergence en cours de clip) et en insérant plusieurs
segments de parole à niveaux (SNR) indépendamment tirés par clip positif
(`generate_vad_sanity_dataset.py`, SNR élargi de 5-15dB fixe à -5-20dB, 1 à 4 segments non
chevauchants au lieu d'un seul) - une diversité de niveau relatif plus directe que ce que la
normalisation RMS + le jitter par clip fournissent seuls.

Alternative envisagée puis écartée : supprimer entièrement la normalisation RMS et compter sur
la seule diversité des données. Gardée malgré tout, car elle limite la plage dynamique réelle en
entrée de la quantification INT8 (§8) - sans elle, l'échelle fixe de quantification devrait
couvrir une plage bien plus large, réduisant la précision effective sur les niveaux "typiques"
et risquant de noyer les signaux faibles/lointains dans le bruit de quantification.

### 14.4 Choisir la bonne métrique : précision à rappel élevé, pas l'AP brute

L'AP (average precision) moyenne la précision sur **tous** les niveaux de rappel, y compris des
niveaux bien en dessous de ce qu'un détecteur utilisable viserait jamais en pratique. Convention
retenue pour ce projet : juger un modèle par sa **précision au seuil qui atteint un rappel
≥95%** - reflète directement le compromis réel ("si je veux capter ~tout, combien de faux
positifs ça coûte ?").

Piège pratique : le seuil correspondant à rappel≥95% est très différent au niveau clip et au
niveau frame (mesuré : 0.9546 vs 0.0561, sur le même modèle) - normal, un clip de plusieurs
secondes n'a besoin que d'**un seul** frame positif pour être positif au niveau clip, donc le
seuil par frame doit être bien plus permissif pour ne rater aucun clip. Comme le firmware décide
en temps réel **par buffer** (jamais par clip agrégé), c'est la courbe précision-rappel
**au niveau frame** qui doit servir à calibrer `TRIGGER_LOGIT_THRESHOLD` (`main.c`) - utiliser le
seuil clip-level donnerait un déclenchement bien plus strict que prévu.

### 14.5 Le bug de parité le plus grave du projet : le gain EMA appliqué avant PCEN à l'entraînement

Découvert après coup, en comparant le comportement du modèle déployé sur de vrais fichiers
audio de test contre les résultats obtenus sur les clips synthétiques (LibriSpeech + FSD50K) du
jeu de test - nettement meilleurs sur les seconds. Les tout premiers LSTM entraînés (sans PCEN,
sans entraînement fenêtré) généralisaient *mieux* sur de vrais enregistrements AudioMoth malgré
l'absence de toutes les optimisations ajoutées depuis - un signal fort qu'une régression de
parité training/déploiement, pas juste un écart synthétique-vs-réel ordinaire, s'était glissée
quelque part.

Cause : `train_student.py::DistillationDataset` a toujours correctement exclu le gain RMS/AGC
(§4) en dehors du mode `"ema"` - avec sa propre justification documentée : appliquer ce gain
waveform-domain avant PCEN double-normalise, et PCEN a besoin de la vraie plage dynamique du
waveform pour calculer des statistiques par bin significatives (voir §6). Le vrai firmware est
d'accord : `computeBufferGain()` retourne un `1.0f` fixe en mode PCEN, aucun gain waveform-domain
du tout. Mais `train_windowed.py::_prepare_clip_waveform` (utilisé pour entraîner **tous** les
checkpoints du sweep d'architectures LSTM/ViT - h16/h32/h64/vit_e32/e48/e64) appliquait ce même
gain **inconditionnellement**, sans jamais vérifier `NORMALIZATION_MODE` - un oubli lors de
l'ajout du support PCEN à ce script, pas un choix délibéré.

Conséquence : chaque checkpoint PCEN entraîné via `train_windowed.py` a vu, à l'entraînement, un
waveform déjà nivelé par l'AGC avant que PCEN ne le voie - mais à l'inférence réelle
(`predict_timeline.py`, le firmware, un vrai AudioMoth), PCEN reçoit toujours le waveform brut,
jamais nivelé. Vérifié directement sur un enregistrement réel (`test3.wav`) avec le checkpoint
`lstm_16_32_64_h64_pcen` : chemin d'inférence réel (brut -> PCEN) donne médiane P(positif)=0.055,
9.5% des frames >0.5 ; chemin correspondant à l'entraînement (nivelé AGC -> PCEN, le bug) donne
médiane=0.292, 28.9% des frames >0.5 - corrélation seulement 0.75 entre les deux. Le modèle a
appris une calibration adaptée à une distribution d'entrée que l'inférence réelle ne lui présente
jamais.

**Corrigé** en ajoutant le même garde-fou `if NORMALIZATION_MODE == "ema":` que
`DistillationDataset` a toujours eu. Aucun réexport ne peut réparer un checkpoint déjà entraîné
avec ce bug - **tous** les checkpoints PCEN existants issus de `train_windowed.py` doivent être
réentraînés sous le code corrigé avant d'être crus sur de l'audio réel.

**Leçon générale, qui rejoint celle du §14.1** : deux fonctions différentes qui prétendent faire
"la même préparation d'entrée" pour un même mode de normalisation doivent être vérifiées
identiques explicitement (même garde-fou, même formule) plutôt que supposées synchrones parce
qu'elles se ressemblent ou partagent un commentaire de docstring - la divergence ne casse rien
syntaxiquement, elle dégrade juste silencieusement, et uniquement en dehors du jeu de test
synthétique sur lequel on validait.

---

## 15. Mode d'enregistrement déclenché (`TRIGGERED_RECORDING_MODE`)

Un troisième mode firmware (mutuellement exclusif avec `STFT_DEBUG_MODE`, `main.c`) : écoute en
continu, fait tourner le classifieur à chaque buffer de 64ms via le pipeline partagé
(`computeSpectrogram()`, factorisé pour être réutilisé par les deux modes de debug), et dès que
le logit dépasse `TRIGGER_LOGIT_THRESHOLD`, ouvre un nouveau fichier WAV - avec ~0.75s de
pre-roll (anneau circulaire des derniers buffers, toujours mis à jour, indépendamment de l'état
de déclenchement), reste ouvert tant que le modèle re-déclenche (fenêtre glissante de 5s,
réinitialisée à chaque re-déclenchement), et se ferme une fois le décompte écoulé - un fichier
distinct par événement de détection, contrairement au mécanisme de seuil d'amplitude déjà
présent dans le firmware normal (`makeRecording()`), qui ne garde jamais qu'un seul fichier
continu avec des trous compressés.

Architecture : un anneau de pre-roll (audio brut, toujours écrit) + une file d'attente
producteur (ISR)/consommateur (boucle principale) d'enregistrements étiquetés par événement
(`TRIGGER_EVENT_START/CONTINUE/END`) - même patron que `stftRecordQueue` du mode debug, adapté
pour porter de l'audio réel plutôt que des données de diagnostic. Une LED verte fixe (pas
clignotante, contrairement aux flashs d'activité habituels du firmware) reste allumée tant qu'un
fichier est ouvert.

Deux bugs réels trouvés au premier vrai test sur device, tous deux dans la configuration du
microphone (copiée telle quelle de `runSTFTFeasibilityTest()`, un outil de test jamais pensé
pour produire de vrais enregistrements écoutables) :

- **Gain fixe.** `AudioMoth_enableMicrophone(AM_NORMAL_GAIN_RANGE, AM_GAIN_MEDIUM, ...)` ignorait
  totalement le gain réellement configuré par l'utilisateur (`configSettings->gain`,
  `enableLowGainRange`) - tout déploiement configuré au-dessus de "medium" sonnait plus
  faiblement que prévu. Corrigé en lisant `configSettings->gain`/`enableLowGainRange`, comme le
  fait `makeRecording()` - `clockDivider`/`acquisitionCycles`/`oversampleRate` restent fixés (ils
  affectent ce que voit le modèle, contrairement au gain, qui n'affecte que le volume).
- **Facteur d'échelle ×16 manquant, effet bien plus important.** `makeRecording()` n'écrit
  jamais d'échantillons ADC bruts dans un WAV directement - chaque échantillon passe par
  `DigitalFilter_applyFilter()`, qui inclut un gain supplémentaire
  (`sampleMultiplier = 16.0f / (oversampleRate * sampleRateDivider)`) compensant le fait que la
  lecture ADC brute sans suréchantillonnage matériel (`oversampleRate=1`, celle utilisée ici pour
  rester fidèle à ce sur quoi le modèle a été entraîné) n'utilise qu'une fraction de la
  convention pleine échelle int16 standard. `TRIGGERED_RECORDING_MODE` échantillonne directement
  via DMA sans jamais appeler `DigitalFilter_applyFilter()` - chaque enregistrement était donc
  ~16x (~24dB) plus faible qu'il n'aurait dû l'être. Corrigé en appliquant ce même facteur
  (`copyScaledAudioForWav()`) **uniquement** aux copies destinées au fichier WAV - jamais à ce
  qui nourrit `computeSpectrogram()`/`ModelInference_run()`, dont la calibration dépend de
  l'échelle ADC brute non modifiée sur laquelle le modèle a été entraîné et validé (voir §11).

Vérification du budget temps réel après ajout de ce mode (même discipline que §12, mesuré sur
device réel, jamais estimé) : ~59% du budget 64ms au repos, ~71% dans le pire cas (rafale de
flush du pre-roll au moment d'un déclenchement) - marge confortable, aucune optimisation requise.

### Checkpoint actuellement déployé et calibration du seuil

`model.h`/`model.c`/`pcen.h`/`pcen.c` embarquent aujourd'hui `student_windowed_lstm_
lstm_16_32_64_h64_pcen.pt` (canaux 16/32/64, LSTM hidden=64 - `MODEL_HIDDEN_SIZE` dans
`model.h`, voir §7 pour l'architecture générique), exporté avec `export_int8.py --long-clips
--calibration-percentile 99.9` (§6). `TRIGGER_PROBABILITY_THRESHOLD` (`main.c`) n'est **pas**
une constante universelle - c'est la précision-à-rappel≥95% (§14.4) mesurée sur l'export du
checkpoint en cours, à re-mesurer et remplacer à chaque nouvel export d'un checkpoint différent
(voir le commentaire au-dessus de la constante dans `main.c`, et l'historique de ses valeurs
successives dans `EXPERIMENTS.md`).

Validation device-vs-simulateur pour ce mode : basculer temporairement `STFT_DEBUG_MODE=1`/
`STFT_OUTPUT_MODE=STFT_OUTPUT_MODE_BINARY`/`TRIGGERED_RECORDING_MODE=0` (mutuellement exclusifs,
voir le haut de `main.c`) pour capturer un `STFT_DATA_*.BIN` contenant le logit réel du modèle
par buffer, puis `compare_device_output.py --long-clips` (§6) pour le rejouer contre
`int8_sim.py` - désactiver aussi `STFT_DEBUG_ATTENTION_TEST` pendant cette capture (son propre
surcoût par buffer risque de faire déborder le budget 64ms et de perdre des buffers, ce qui
casserait la continuité de l'état LSTM que cette comparaison exige).

---

## 16. Récapitulatif des bugs rencontrés (pour check-list de non-régression)

1. **Dispatcher FFT générique** gonfle la flash de ~75KB (tables jamais utilisées) - corrigé en
   appelant `arm_rfft_fast_init_256_f32()` directement.
2. **STFT continue vs par buffer** - une frame de trop calculée à chaque frontière de buffer
   côté entraînement - corrigé en restructurant `LogSpectrogram` pour redémarrer le comptage de
   hops à chaque buffer, comme le firmware.
3. **Échelle brute int16 vs `[-1,1]`** - écart constant de ~90.3dB - corrigé en chargeant
   l'audio d'entraînement en `dtype="int16"`.
4. **Normalisation RMS manquante côté firmware** - corrigé par un gain EMA glissant, répliqué
   à l'identique côté entraînement (`streaming_rms_normalize`).
5. **Filtre passe-haut manquant côté firmware** - corrigé en masquant les bins < 150Hz. Retiré
   des deux côtés (entraînement + firmware) puis réintégré plus tard dans le même projet - voir
   bug 16 ci-dessous et `EXPERIMENTS.md` pour l'historique complet de ce retour en arrière.
6. **`runAvgPoolFreq()` : indexation NCHW au lieu de NHWC** - corrigé, confirmé par test
   numérique (erreur max 29/127 avant, 0 après).
7. **Layout des poids depthwise à l'export** - PyTorch `[Cout,1,kH,kW]` écrit tel quel au lieu
   du layout CMSIS-NN `[1,kH,kW,Cout]` - corrigé par une permutation à l'export (367/648 sorties
   fausses avant, 0/648 après).
8. **Jitter de calibration non déterministe** - tiré du RNG global au lieu d'un générateur
   seedé par index - rendait la calibration/export non reproductible d'un run à l'autre.
9. **LSTM `effective_bias` NULL (bug majeur, §11.3)** - le kernel réel ne lit jamais `bias`,
   seulement `input_effective_bias`/`hidden_effective_bias`, laissés `NULL` -> lecture
   silencieuse de la table de vecteurs d'interruption comme "biais". Corrigé en calculant et
   exportant `input_effective_bias` (et un tableau de zéros pour `hidden_effective_bias`).
   Impact mesuré : corrélation device/PC 0.317 -> 0.9999, écart moyen 0.474 -> 0.0012.
10. **CNN continu vs relancé par buffer à l'entraînement (bug majeur, §14.1)** - `frame_logits()`
    et les deux réimplémentations d'`export_int8.py` calculaient le CNN en continu sur tout un
    clip, donnant un champ réceptif que le firmware n'a jamais en réalité. Corrigé en relançant
    le CNN par buffer partout, avec un test de cohérence entre les implémentations qui n'existait
    pas avant.
11. **Constante de temps de l'EMA RMS dérivée de `CLIP_SECONDS` (§14.2)** - aurait silencieusement
    désynchronisé l'entraînement du comportement réel du firmware dès qu'on change la durée des
    clips d'entraînement. Corrigée en la fixant à la constante réelle du firmware (46.875
    buffers), indépendamment de `CLIP_SECONDS`.
12. **`TRIGGERED_RECORDING_MODE` : gain microphone fixe (§15)** - `AM_GAIN_MEDIUM` codé en dur au
    lieu de `configSettings->gain`, copié d'un outil de test jamais pensé pour produire de vrais
    enregistrements.
13. **`TRIGGERED_RECORDING_MODE` : facteur d'échelle ×16 manquant sur l'audio WAV (§15)** -
    l'ADC en lecture brute (`oversampleRate=1`) produit des échantillons ~16x plus petits que la
    convention pleine échelle habituelle ; le firmware normal compense via
    `DigitalFilter_applyFilter()`, que ce mode ne traverse jamais. Corrigé en appliquant ce
    facteur uniquement sur les copies destinées au WAV, jamais sur ce qui nourrit le modèle.
14. **Calibration INT8 sur le max littéral, PCEN (§6)** - PCEN a une distribution de sortie à
    queue très lourde ("attack overshoot" transitoire quand `M` n'a pas rattrapé un son fort qui
    vient de démarrer) : le max littéral gaspille presque toute la résolution INT8 sur des
    valeurs qui n'arrivent presque jamais. AP frame INT8 87.1% (calibration sur le max) contre
    96.0% (calibration sur le p99.9), pour le même checkpoint. Corrigé en ajoutant
    `--calibration-percentile` à `export_int8.py`/`export_pcen.py` (défaut 100 = comportement
    inchangé, à ajuster explicitement en mode PCEN).
15. **Gain EMA appliqué avant PCEN à l'entraînement, jamais à l'inférence réelle (bug le plus
    grave du projet, §14.5)** - `train_windowed.py::_prepare_clip_waveform` appliquait le gain
    RMS/AGC inconditionnellement, sans jamais vérifier `NORMALIZATION_MODE`, contrairement à
    `DistillationDataset` qui l'a toujours correctement exclu hors du mode `"ema"`. Tous les
    checkpoints PCEN issus de `train_windowed.py` (le sweep complet h16/h32/h64/vit_e32/e48/e64)
    ont été entraînés sur un waveform déjà nivelé par l'AGC avant PCEN, jamais présenté à
    l'inférence réelle - corrélation mesurée seulement 0.75 entre les deux chemins sur un vrai
    enregistrement. Corrigé en ajoutant le même garde-fou que `DistillationDataset`. Aucun
    réexport ne répare un checkpoint déjà entraîné avec ce bug - retraining obligatoire.

**Test de non-régression le plus utile en cas de doute** : pour les bugs 1-9 (parité numérique
device-vs-PC), relancer `compare_device_output.py` sur une capture fraîche - un écart device/PC
qui redevient grand et/ou faiblement corrélé signale presque toujours l'un d'eux réapparu. Pour
les bugs 10-11 (parité entraînement/déploiement), vérifier que `model.frame_logits(spec)` et
`float_forward_with_traces(spec, folded, model)["logits"]` concordent toujours (§14.1), et que
`dataset.RMS_EMA_ALPHA` ne change jamais quand `CLIP_SECONDS` change (§14.2). Pour le bug 14,
vérifier qu'`export_int8.py --long-clips` avec `--calibration-percentile 100` (le max) donne un
AP INT8 nettement pire qu'avec `99.9` sur un checkpoint PCEN - sinon la distribution a changé et
le percentile mérite d'être réévalué. Pour le bug 15 (le plus insidieux : rien ne plante, la
dégradation n'apparaît que sur de l'audio réel, jamais sur le jeu de test synthétique), vérifier
directement que `_prepare_clip_waveform`/`DistillationDataset` ont bien le même garde-fou
`NORMALIZATION_MODE` avant de faire confiance à un nouveau checkpoint PCEN sur le terrain.
