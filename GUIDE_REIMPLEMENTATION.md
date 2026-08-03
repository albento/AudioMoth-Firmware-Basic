# Guide de réimplémentation : STFT + classifieur INT8 (CMSIS-DSP/CMSIS-NN) sur AudioMoth

Ce document rassemble tout ce qu'il faut savoir pour ré-implémenter, seul et en comprenant
chaque brique, le pipeline actuel : capture audio -> STFT -> normalisation -> modèle
`TinyLSTMCNN` quantifié en INT8 -> classification, sur le AudioMoth Dev
(EFM32WG380F256, Cortex-M4F @ 48MHz, 256KB flash, 32KB RAM interne + 256KB SRAM externe).

Il part du principe que vous ne connaissez pas CMSIS-DSP/CMSIS-NN, et détaille en particulier
les optimisations de convolution et les problèmes rencontrés avec la couche LSTM, qui ont été
la source du bug le plus sérieux de toute l'investigation.

Sources citées : `src/main.c`, `src/model_inference.c` (ce dépôt), `export_int8.py`,
`int8_sim.py`, `student_model.py`, `dataset.py` (`test-audiomoth-cnn`), et le code source de
CMSIS-NN lui-même (`AudioMoth-Project/CMSIS-NN/Source/...`).

---

## 1. Vue d'ensemble du pipeline

```
Microphone (ADC, 16kHz, int16 brut)
  -> buffer de 1024 échantillons (64ms), un par interruption DMA
  -> gain RMS glissant (normalisation causale)
  -> fenêtrage de Hann + FFT 256 points -> magnitude -> log10 (approx. fixe)
  -> masquage des bins < 150Hz (passe-haut)
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

## 6. Le modèle : TinyLSTMCNN

Architecture (`student_model.py::TinyLSTMCNN`, canaux 16/32/64, LSTM hidden=16) :

| Étage | Type | in_ch | out_ch | kernel | stride | pad | groups |
|---|---|---|---|---|---|---|---|
| stage0 | conv classique | 1 | 16 | 3x3 | 2 | 1 | 1 |
| stage1_dw | depthwise | 16 | 16 | 3x3 | 2 | 1 | 16 |
| stage1_pw | pointwise 1x1 | 16 | 32 | 1x1 | 1 | 0 | 1 |
| stage2_dw | depthwise | 32 | 32 | 3x3 | 2 | 1 | 32 |
| stage2_pw | pointwise 1x1 | 32 | 64 | 1x1 | 1 | 0 | 1 |

Chaque conv est suivie de BatchNorm (fusionné dans les poids avant quantification, voir §8) et
ReLU. Avec l'entrée `[1, 129, 7]` (fréquence x temps), les 3 étages à stride 2 réduisent le
temps `7 -> 4 -> 2 -> 1` : **un seul pas de temps en sortie par buffer**.

Ensuite : `AdaptiveAvgPool2d((1, None))` (moyenne sur l'axe fréquence uniquement, garde le
temps) -> LSTM à une couche (`nn.LSTM(64, 16, batch_first=True)`) -> classifieur linéaire
(`nn.Linear(16, 1)`).

---

## 7. Quantification INT8 : les notions de base

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
voir §10 pour le cas particulier - et piégeux - du LSTM).

---

## 8. CMSIS-NN : conventions et fonctions de convolution

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
  multiplicateur/shift par canal de sortie (voir §7.3).

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
explicitement (voir §11.3 pour le code). Ne faites jamais ça si votre modèle a un
`input_offset` non nul (asymétrique) : il faudrait alors ajouter `input_offset * poids` pour
chaque tap manquant, ce qui annule tout le bénéfice.

Pour les 2 étages "pointwise" (1x1, stride 1, pad 0), im2col ne fait presque rien (une fenêtre
1x1 = juste le pixel lui-même, pas de gather réel) donc `arm_convolve_s8` reste la meilleure
option mesurée ici - voir §11 pour la tentative avortée de le remplacer.

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

## 9. Organisation mémoire des tenseurs d'activation

Le plus gros tenseur intermédiaire (sortie de `stage0`, `16 canaux * 65 * 4 = 4160 octets`) ne
rentre pas dans la RAM interne déjà largement occupée par le reste du firmware (~4-5KB
restants). Solution : deux buffers "ping-pong" (`activationBufferA`/`B`) placés en SRAM externe
(lente, mais avec largement la place), chaque étage écrivant dans l'un en lisant l'autre, en
alternance. Le tenseur d'entrée quantifié et le petit vecteur pooled (entrée du LSTM, 64
valeurs) restent en RAM interne (rapide, petits).

---

## 10. Les couches LSTM CMSIS-NN : LE point sensible

C'est ici qu'a eu lieu le bug le plus sérieux de toute l'investigation (biais silencieusement
faux sur chaque pas de LSTM, chaque buffer) - cette section mérite d'être lue attentivement
avant toute réimplémentation.

### 10.1 Vue d'ensemble : structures et convention de format

Un LSTM a 4 "portes" (forget, input, cell, output), chacune calculée comme :
`porte = activation(x @ W_input.T + h_prev @ W_hidden.T + bias)`, où `activation` est sigmoïde
(forget/input/output) ou tanh (cell). CMSIS-NN représente ça avec :

- `cmsis_nn_lstm_gate` (une instance par porte) : poids input/hidden (int8), multiplicateurs/
  shifts (scalaires, **quantification par tenseur**, voir §7.2), type d'activation, et - point
  crucial - `bias` **et** `input_effective_bias`/`hidden_effective_bias` (voir §10.3).
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
vérification sur device dédié pour être détecté (voir §12).

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
bug du §10.3, contre 22-26/32 avant). Ne cherchez pas à éliminer ce bruit résiduel si vous
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

Ces deux facteurs réels passent chacun par `quantize_multiplier()` (§7.3) pour donner
`forget_to_cell_multiplier/shift` et `input_to_cell_multiplier/shift`. Voir
`int8_sim.py::cell_multipliers()` pour l'implémentation canonique (partagée entre le
simulateur et l'export C, pour être sûr que les deux ne divergent jamais).

---

## 11. Optimisations de performance mesurées

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

## 12. Méthodologie de validation (comment vérifier chaque étape sans se tromper)

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
   test qui a permis de détecter/diagnostiquer le bug du §10.3 (corrélation device/PC 0.32 avant
   correction, 0.9999 après).

---

## 13. Récapitulatif des bugs rencontrés (pour check-list de non-régression)

1. **Dispatcher FFT générique** gonfle la flash de ~75KB (tables jamais utilisées) - corrigé en
   appelant `arm_rfft_fast_init_256_f32()` directement.
2. **STFT continue vs par buffer** - une frame de trop calculée à chaque frontière de buffer
   côté entraînement - corrigé en restructurant `LogSpectrogram` pour redémarrer le comptage de
   hops à chaque buffer, comme le firmware.
3. **Échelle brute int16 vs `[-1,1]`** - écart constant de ~90.3dB - corrigé en chargeant
   l'audio d'entraînement en `dtype="int16"`.
4. **Normalisation RMS manquante côté firmware** - corrigé par un gain EMA glissant, répliqué
   à l'identique côté entraînement (`streaming_rms_normalize`).
5. **Filtre passe-haut manquant côté firmware** - corrigé en masquant les bins < 150Hz.
6. **`runAvgPoolFreq()` : indexation NCHW au lieu de NHWC** - corrigé, confirmé par test
   numérique (erreur max 29/127 avant, 0 après).
7. **Layout des poids depthwise à l'export** - PyTorch `[Cout,1,kH,kW]` écrit tel quel au lieu
   du layout CMSIS-NN `[1,kH,kW,Cout]` - corrigé par une permutation à l'export (367/648 sorties
   fausses avant, 0/648 après).
8. **Jitter de calibration non déterministe** - tiré du RNG global au lieu d'un générateur
   seedé par index - rendait la calibration/export non reproductible d'un run à l'autre.
9. **LSTM `effective_bias` NULL (bug majeur, §10.3)** - le kernel réel ne lit jamais `bias`,
   seulement `input_effective_bias`/`hidden_effective_bias`, laissés `NULL` -> lecture
   silencieuse de la table de vecteurs d'interruption comme "biais". Corrigé en calculant et
   exportant `input_effective_bias` (et un tableau de zéros pour `hidden_effective_bias`).
   Impact mesuré : corrélation device/PC 0.317 -> 0.9999, écart moyen 0.474 -> 0.0012.

**Test de non-régression le plus utile en cas de doute** : relancer
`compare_device_output.py` sur une capture fraîche - un écart device/PC qui redevient grand
et/ou faiblement corrélé signale presque toujours un de ces neuf bugs (ou un nouveau du même
genre) réapparu.
