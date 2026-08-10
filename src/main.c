/****************************************************************************
 * main.c
 * openacousticdevices.info
 * June 2017
 *****************************************************************************/

#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "gps.h"
#include "sunrise.h"
#include "audiomoth.h"
#include "audioconfig.h"
#include "digitalfilter.h"
#include "model_inference.h"
#include "pcen.h"

/* STFT feasibility debug mode - set to 0 to restore normal recording firmware. Mutually
   exclusive with TRIGGERED_RECORDING_MODE below - only one of the two may be 1 at a time
   (see the #error guard just below). */

#define STFT_DEBUG_MODE                          1

/* Continuously listens and runs the on-device INT8 classifier (model_inference.c) every
   64ms buffer; whenever its output crosses TRIGGER_PROBABILITY_THRESHOLD, records a new WAV
   file (with ~TRIGGER_PREROLL_SECONDS of pre-roll audio), keeping it open while re-triggered
   (sliding TRIGGER_TAIL_SECONDS countdown) and closing it once the countdown lapses - see
   runTriggeredRecording(). See GUIDE_REIMPLEMENTATION.md for the STFT/model pipeline this
   reuses, and the plan this was built from for the full design rationale. */

#define TRIGGERED_RECORDING_MODE                 0

#if STFT_DEBUG_MODE && TRIGGERED_RECORDING_MODE
#error "STFT_DEBUG_MODE and TRIGGERED_RECORDING_MODE are mutually exclusive - set at most one to 1"
#endif

/* Only meaningful alongside STFT_DEBUG_MODE=1 (compiles to nothing otherwise) - runs an
   isolated, dummy-weight ALiBi windowed-attention encoder layer (see attnRunLayer() further
   below) once per raw STFT frame, purely to get a real hardware timing number for whether a
   ViT could plausibly run in real time here - NOT a step towards actually deploying it: no
   trained/exported ViT weights exist, model_inference.c/model.h/the LSTM classifier are
   completely untouched by this, in this mode and every other. See
   test-audiomoth-cnn/student_model.py's _AlibiEncoderLayer/_AlibiSelfAttention for the
   reference this ports, and this section's own comment further below for the full design
   rationale.

   Set to 0 here (test-audiomoth-cnn/compare_device_output.py capture): this test's whole
   point is a clean, gap-free BINARY capture of the real classifier's own logit, carried
   continuously buffer to buffer - the attention probe's own extra per-buffer work (already
   ~1.18x the 64ms budget on its own, per this session's own design discussion) risks DMA
   buffer overruns/dropped buffers, which would break the LSTM state continuity
   compare_device_output.py depends on. */

#define STFT_DEBUG_ATTENTION_TEST                0

#if STFT_DEBUG_MODE || TRIGGERED_RECORDING_MODE
#include "arm_math.h"
#endif

#if STFT_DEBUG_MODE && STFT_DEBUG_ATTENTION_TEST
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#endif

/* Useful time constants */

#define MICROSECONDS_IN_SECOND                  1000000
#define MICROSECONDS_IN_MILLISECOND             1000
#define MILLISECONDS_IN_SECOND                  1000

#define SECONDS_IN_MINUTE                       60
#define SECONDS_IN_HOUR                         (60 * SECONDS_IN_MINUTE)
#define SECONDS_IN_DAY                          (24 * SECONDS_IN_HOUR)

#define MINUTES_IN_HOUR                         60
#define MINUTES_IN_DAY                          1440
#define YEAR_OFFSET                             1900
#define MONTH_OFFSET                            1              

#define START_OF_CENTURY                        946684800
#define MIDPOINT_OF_CENTURY                     2524608000

/* Useful coordinate constant */

#define MINUTES_IN_DEGREE                       60

/* Useful type constants */

#define BITS_PER_BYTE                           8
#define UINT32_SIZE_IN_BITS                     32
#define UINT32_SIZE_IN_BYTES                    4
#define UINT16_SIZE_IN_BYTES                    2

/* Sleep and LED constants */

#define LOW_BATTERY_LED_FLASHES                 10

#define SHORT_LED_FLASH_DURATION                100
#define LONG_LED_FLASH_DURATION                 500

#define TRIGGERED_RECORDING_FLASH_DURATION      4

#define WAITING_LED_FLASH_DURATION              10
#define WAITING_LED_FLASH_INTERVAL              2000

#define MINIMUM_LED_FLASH_INTERVAL              500

#define SHORT_WAIT_INTERVAL                     100
#define DEFAULT_WAIT_INTERVAL                   1000

#define USB_CONFIGURATION_BLINK                 400

/* SRAM buffer constants */

#define NUMBER_OF_BUFFERS                       8
#define NUMBER_OF_BYTES_IN_SAMPLE               2
#define EXTERNAL_SRAM_SIZE_IN_SAMPLES           (AM_EXTERNAL_SRAM_SIZE_IN_BYTES / NUMBER_OF_BYTES_IN_SAMPLE)
#define NUMBER_OF_SAMPLES_IN_BUFFER             (EXTERNAL_SRAM_SIZE_IN_SAMPLES / NUMBER_OF_BUFFERS)

/* DMA transfer constant */

#define MAXIMUM_SAMPLES_IN_DMA_TRANSFER         1024

/* Compression constant */

#define COMPRESSION_BUFFER_SIZE_IN_BYTES        512

/* File size constants */

#define MAXIMUM_FILE_NAME_LENGTH                64

#define MAXIMUM_WAV_DATA_SIZE                   (UINT32_MAX - 16 * 1024 * 1024)

/* Configuration file constants */

#define CONFIG_BUFFER_LENGTH                    512
#define CONFIG_TIMEZONE_LENGTH                  8

/* WAV header constants */

#define PCM_FORMAT                              1
#define RIFF_ID_LENGTH                          4
#define LENGTH_OF_ARTIST                        32
#define LENGTH_OF_COMMENT                       384

/* USB configuration constant */

#define MAX_RECORDING_PERIODS                   5

/* Digital filter constant */

#define FILTER_FREQ_MULTIPLIER                  100

/* DC filter constants */

#define LOW_DC_BLOCKING_FREQ                    8
#define DEFAULT_DC_BLOCKING_FREQ                48

/* Supply voltage constant */

#define MINIMUM_SUPPLY_VOLTAGE                  2800

/* Consecutive recording error constant */

#define MAX_CONSECUTIVE_RECORDING_ERRORS        5

/* Deployment ID constant */

#define DEPLOYMENT_ID_LENGTH                    8

/* Acoustic location constant */

#define ACOUSTIC_LOCATION_SIZE_IN_BYTES         7

/* Audio configuration constants */

#define AUDIO_CONFIG_PULSE_INTERVAL             10
#define AUDIO_CONFIG_TIME_CORRECTION            134
#define AUDIO_CONFIG_TONE_TIMEOUT               250
#define AUDIO_CONFIG_PACKETS_TIMEOUT            30000

/* GPS time setting constants */

#define GPS_MAXIMUM_DIFFERENCE                  SECONDS_IN_HOUR
#define GPS_INITIAL_TIME_SETTING_PERIOD         300
#define GPS_DEFAULT_TIME_SETTING_PERIOD         300
#define GPS_MIN_TIME_SETTING_PERIOD             30
#define GPS_ADDITIONAL_POWER_INTERVAL           2
#define GPS_TIME_SETTING_MARGIN                 2
#define GPS_FREQUENCY_PRECISION                 1000
#define GPS_MESSAGE_BUFFER_SIZE_IN_BYTES        128
#define GPS_FILENAME                            "GPS.TXT"

/* Magnetic switch constants */

#define MAGNETIC_SWITCH_WAIT_MULTIPLIER         2
#define MAGNETIC_SWITCH_CHANGE_FLASHES          10

/* USB configuration constant */

#define USB_CONFIG_TIME_CORRECTION              26

/* Recording start time correction */

#define DELAY_CORRECTION_MICROSECONDS           80

/* Recording preparation constants */

#define MINIMUM_PREPARATION_PERIOD              500
#define INITIAL_PREPARATION_PERIOD              1000
#define MAXIMUM_PREPARATION_PERIOD              30000

#define DEFAULT_PREPARATION_PERIOD_BUFFER       2000

#define PREPARATION_PERIOD_SMOOTHING_FACTOR     4
#define PREPARATION_PERIOD_LED_DIVIDER          4

/* Energy saver mode constant */

#define ENERGY_SAVER_SAMPLE_RATE_THRESHOLD      48000

/* Frequency trigger constants */

#define FREQUENCY_TRIGGER_WINDOW_MINIMUM        16
#define FREQUENCY_TRIGGER_WINDOW_MAXIMUM        1024

/* Sunrise and sunset recording constants */

#define MINIMUM_SUN_RECORDING_GAP               60
#define SUN_RECORDING_GAP_MULTIPLIER            4

/* Location constants */

#define ACOUSTIC_LONGITUDE_MULTIPLIER           2

#define CONFIG_LOCATION_PRECISION               100
#define ACOUSTIC_LOCATION_PRECISION             1000000
#define GPS_LOCATION_PRECISION                  1000000

/* Useful macros */

#define FLASH_LED(led, duration) { \
    AudioMoth_set ## led ## LED(true); \
    AudioMoth_delay(duration); \
    AudioMoth_set ## led ## LED(false); \
}

#define FLASH_REPEAT_LED(led, repeats, duration) { \
    for (uint32_t i = 0; i < repeats; i += 1) { \
        AudioMoth_set ## led ## LED(true); \
        AudioMoth_delay(duration); \
        AudioMoth_set ## led ## LED(false); \
        AudioMoth_delay(duration); \
    } \
}

#define FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(fn) { \
    bool success = (fn); \
    if (success != true) { \
        if (enableLED) { \
            AudioMoth_setBothLED(false); \
            AudioMoth_delay(LONG_LED_FLASH_DURATION); \
            FLASH_LED(Both, LONG_LED_FLASH_DURATION) \
        } \
        return SDCARD_WRITE_ERROR; \
    } \
}

#define RETURN_BOOL_ON_ERROR(fn) { \
    bool success = (fn); \
    if (success != true) { \
        return success; \
    } \
}

#define SAVE_SWITCH_POSITION_AND_POWER_DOWN(milliseconds) { \
    *previousSwitchPosition = switchPosition; \
    AudioMoth_powerDownAndWakeMilliseconds(milliseconds); \
}

#define SERIAL_NUMBER                           "%08X%08X"

#define FORMAT_SERIAL_NUMBER(src)               (unsigned int)*((uint32_t*)src + 1),  (unsigned int)*((uint32_t*)src)

#define ABS(a)                                  ((a) < (0) ? (-a) : (a))

#define MIN(a, b)                               ((a) < (b) ? (a) : (b))

#define MAX(a, b)                               ((a) > (b) ? (a) : (b))

#define ROUNDED_DIV(a, b)                       (((a) + ((b)/2)) / (b))

#define ROUNDED_UP_DIV(a, b)                    (((a) + (b) - 1) / (b))

#define ROUND_UP_TO_MULTIPLE(a, b)              (((a) + (b) - 1) & ~((b)-1))

#define UNSIGNED_ROUND(n, d)                    ((d) * (((n) + (d)/2) / (d)))

/* Recording state enumeration */

typedef enum {RECORDING_OKAY, FILE_SIZE_LIMITED, SUPPLY_VOLTAGE_LOW, SWITCH_CHANGED, MICROPHONE_CHANGED, MAGNETIC_SWITCH, SDCARD_WRITE_ERROR} AM_recordingState_t;

/* Filter type enumeration */

typedef enum {NO_FILTER, LOW_PASS_FILTER, BAND_PASS_FILTER, HIGH_PASS_FILTER} AM_filterType_t;

/* Battery level display type */

typedef enum {BATTERY_LEVEL, NIMH_LIPO_BATTERY_VOLTAGE} AM_batteryLevelDisplayType_t;

/* Sun recording mode enumeration */

typedef enum {SUNRISE_RECORDING, SUNSET_RECORDING, SUNRISE_AND_SUNSET_RECORDING, SUNSET_TO_SUNRISE_RECORDING, SUNRISE_TO_SUNSET_RECORDING} AM_sunRecordingMode_t;

/* Sun recording mode enumeration */

typedef enum {INITIAL_GPS_FIX, GPS_FIX_BEFORE_RECORDING_PERIOD, GPS_FIX_BETWEEN_RECORDING_PERIODS, GPS_FIX_AFTER_RECORDING_PERIOD, GPS_FIX_BEFORE_INDIVIDUAL_RECORDING, GPS_FIX_BETWEEN_INDIVIDUAL_RECORDINGS, GPS_FIX_AFTER_INDIVIDUAL_RECORDING} AM_gpsFixMode_t;

/* WAV header */

#pragma pack(push, 1)

typedef struct {
    char id[RIFF_ID_LENGTH];
    uint32_t size;
} chunk_t;

typedef struct {
    chunk_t icmt;
    char comment[LENGTH_OF_COMMENT];
} icmt_t;

typedef struct {
    chunk_t iart;
    char artist[LENGTH_OF_ARTIST];
} iart_t;

typedef struct {
    uint16_t format;
    uint16_t numberOfChannels;
    uint32_t samplesPerSecond;
    uint32_t bytesPerSecond;
    uint16_t bytesPerCapture;
    uint16_t bitsPerSample;
} wavFormat_t;

typedef struct {
    chunk_t riff;
    char format[RIFF_ID_LENGTH];
    chunk_t fmt;
    wavFormat_t wavFormat;
    chunk_t list;
    char info[RIFF_ID_LENGTH];
    icmt_t icmt;
    iart_t iart;
    chunk_t data;
} wavHeader_t;

#pragma pack(pop)

static wavHeader_t wavHeader = {
    .riff = {.id = "RIFF", .size = 0},
    .format = "WAVE",
    .fmt = {.id = "fmt ", .size = sizeof(wavFormat_t)},
    .wavFormat = {.format = PCM_FORMAT, .numberOfChannels = 1, .samplesPerSecond = 0, .bytesPerSecond = 0, .bytesPerCapture = 2, .bitsPerSample = 16},
    .list = {.id = "LIST", .size = RIFF_ID_LENGTH + sizeof(icmt_t) + sizeof(iart_t)},
    .info = "INFO",
    .icmt = {.icmt.id = "ICMT", .icmt.size = LENGTH_OF_COMMENT, .comment = ""},
    .iart = {.iart.id = "IART", .iart.size = LENGTH_OF_ARTIST, .artist = ""},
    .data = {.id = "data", .size = 0}
};

/* USB configuration data structure */

#pragma pack(push, 1)

typedef struct {
    uint16_t startMinutes;
    uint16_t endMinutes;
} recordingPeriod_t;

typedef struct {
    uint32_t time;
    AM_gainSetting_t gain;
    uint8_t clockDivider;
    uint8_t acquisitionCycles;
    uint8_t oversampleRate;
    uint32_t sampleRate;
    uint8_t sampleRateDivider;
    uint16_t sleepDuration;
    uint16_t recordDuration;
    uint8_t enableLED;
    union {
        struct {
            uint8_t activeRecordingPeriods;
            recordingPeriod_t recordingPeriods[MAX_RECORDING_PERIODS];
        };
        struct {
            uint8_t sunRecordingMode : 3;
            uint8_t sunRecordingEvent : 2;
            int16_t latitude;
            int16_t longitude;
            uint8_t sunRoundingMinutes;
            uint16_t beforeSunriseMinutes : 10;
            uint16_t afterSunriseMinutes : 10;
            uint16_t beforeSunsetMinutes : 10;
            uint16_t afterSunsetMinutes : 10;
        };
    };
    int8_t timezoneHours;
    uint8_t enableLowVoltageCutoff;
    uint8_t disableBatteryLevelDisplay : 1;
    uint8_t requireAcousticLocation : 1;
    uint8_t useTimezoneFromAcousticChime: 1;
    uint8_t adjustScheduleUsingTimezoneFromAcousticChime: 1;
    uint8_t preparationPeriodBuffer: 4;
    int8_t timezoneMinutes;
    uint8_t disableSleepRecordCycle : 1;
    uint8_t enableFilenameWithDeviceID : 1;
    uint8_t enableTimeSettingBeforeAndAfterRecordings: 1;
    uint8_t gpsTimeSettingPeriod: 4;
    uint8_t ignoreExternalMicrophoneForAcousticChime: 1;
    uint32_t earliestRecordingTime;
    uint32_t latestRecordingTime;
    uint16_t lowerFilterFreq;
    uint16_t higherFilterFreq;
    union {
        uint16_t amplitudeThreshold;
        uint16_t frequencyTriggerCentreFrequency;
    };
    uint8_t requireAcousticConfiguration : 1;
    AM_batteryLevelDisplayType_t batteryLevelDisplayType : 1;
    uint8_t minimumTriggerDuration : 6;
    union {
        struct {
            uint8_t frequencyTriggerWindowLengthShift : 4; 
            uint8_t frequencyTriggerThresholdPercentageMantissa : 4; 
            int8_t frequencyTriggerThresholdPercentageExponent : 3; 
        };
        struct {
            uint8_t enableAmplitudeThresholdDecibelScale : 1;
            uint8_t amplitudeThresholdDecibels : 7; 
            uint8_t enableAmplitudeThresholdPercentageScale : 1;
            uint8_t amplitudeThresholdPercentageMantissa : 4; 
            int8_t amplitudeThresholdPercentageExponent : 3; 
        };
    };
    uint8_t enableEnergySaverMode : 1; 
    uint8_t disable48HzDCBlockingFilter : 1;
    uint8_t enableTimeSettingFromGPS : 1;
    uint8_t enableMagneticSwitch : 1;
    uint8_t enableLowGainRange : 1;
    uint8_t enableFrequencyTrigger : 1;
    uint8_t enableDailyFolders : 1;
    uint8_t enableSunRecording : 1;
} configSettings_t;

#pragma pack(pop)

static const configSettings_t defaultConfigSettings = {
    .time = 0,
    .gain = AM_GAIN_MEDIUM,
    .clockDivider = 4,
    .acquisitionCycles = 16,
    .oversampleRate = 1,
    .sampleRate = 384000,
    .sampleRateDivider = 8,
    .sleepDuration = 5,
    .recordDuration = 55,
    .enableLED = 1,
    .activeRecordingPeriods = 1,
    .recordingPeriods = {
        {.startMinutes = 0, .endMinutes = 0},
        {.startMinutes = 0, .endMinutes = 0},
        {.startMinutes = 0, .endMinutes = 0},
        {.startMinutes = 0, .endMinutes = 0},
        {.startMinutes = 0, .endMinutes = 0}
    },
    .timezoneHours = 0,
    .enableLowVoltageCutoff = 1,
    .disableBatteryLevelDisplay = 0,
    .requireAcousticLocation = 0,
    .useTimezoneFromAcousticChime = 0,
    .adjustScheduleUsingTimezoneFromAcousticChime = 0,
    .preparationPeriodBuffer = 2,
    .timezoneMinutes = 0,
    .disableSleepRecordCycle = 0,
    .enableFilenameWithDeviceID = 0,
    .enableTimeSettingBeforeAndAfterRecordings = 0,
    .gpsTimeSettingPeriod = 0,
    .ignoreExternalMicrophoneForAcousticChime = 0,
    .earliestRecordingTime = 0,
    .latestRecordingTime = 0,
    .lowerFilterFreq = 0,
    .higherFilterFreq = 0,
    .amplitudeThreshold = 0,
    .requireAcousticConfiguration = 0,
    .batteryLevelDisplayType = BATTERY_LEVEL,
    .minimumTriggerDuration = 0,
    .enableAmplitudeThresholdDecibelScale = 0,
    .amplitudeThresholdDecibels = 0,
    .enableAmplitudeThresholdPercentageScale = 0,
    .amplitudeThresholdPercentageMantissa = 0,
    .amplitudeThresholdPercentageExponent = 0,
    .enableEnergySaverMode = 0,
    .disable48HzDCBlockingFilter = 0,
    .enableTimeSettingFromGPS = 0,
    .enableMagneticSwitch = 0,
    .enableLowGainRange = 0,
    .enableFrequencyTrigger = 0,
    .enableDailyFolders = 0,
    .enableSunRecording = 0
};

/* Persistent configuration data structure */

#pragma pack(push, 1)

typedef struct {
    uint8_t firmwareVersion[AM_FIRMWARE_VERSION_LENGTH];
    uint8_t firmwareDescription[AM_FIRMWARE_DESCRIPTION_LENGTH];
    configSettings_t configSettings;
} persistentConfigSettings_t;

#pragma pack(pop)

/* Acoustic location data structure */

#pragma pack(push, 1)

typedef struct {
    int32_t latitude: 28;
    int32_t longitude: 28;
} acousticLocation_t;

#pragma pack(pop)

/* Function to select energy saver mode */

static bool isEnergySaverMode(configSettings_t *configSettings) {

    return configSettings->enableEnergySaverMode && configSettings->sampleRate / configSettings->sampleRateDivider <= ENERGY_SAVER_SAMPLE_RATE_THRESHOLD;

}

/* Functions to format header and configuration components */

static uint32_t formatDecibels(char *dest, uint32_t value, bool space) {

    if (value > 0) return sprintf(dest, space ? "-%lu dB" : "-%ludB", value);

    return sprintf(dest, space ? "0 dB" : "0dB");

}

static uint32_t formatPercentage(char *dest, uint32_t mantissa, int32_t exponent) {

    uint32_t length = exponent < 0 ? 1 - exponent : 0;

    memcpy(dest, "0.0000", length);

    length += sprintf(dest + length, "%lu", mantissa);

    while (exponent-- > 0) dest[length++] = '0';

    dest[length++] = '%';

    return length;

}

/* Functions to set WAV header details and comment */

static void setHeaderDetails(wavHeader_t *wavHeader, uint32_t sampleRate, uint32_t numberOfSamples, uint32_t guanoDataSize) {

    wavHeader->wavFormat.samplesPerSecond = sampleRate;
    wavHeader->wavFormat.bytesPerSecond = NUMBER_OF_BYTES_IN_SAMPLE * sampleRate;
    wavHeader->data.size = NUMBER_OF_BYTES_IN_SAMPLE * numberOfSamples;
    wavHeader->riff.size = NUMBER_OF_BYTES_IN_SAMPLE * numberOfSamples + sizeof(wavHeader_t) + guanoDataSize - sizeof(chunk_t);

}

static void setHeaderComment(wavHeader_t *wavHeader, configSettings_t *configSettings, uint32_t currentTime, uint8_t *serialNumber, uint8_t *deploymentID, uint8_t *defaultDeploymentID, AM_extendedBatteryState_t extendedBatteryState, int32_t temperature, bool externalMicrophone, AM_recordingState_t recordingState, AM_filterType_t filterType) {

    struct tm time;

    int32_t timezoneHours, timezoneMinutes;

    AudioMoth_timezoneRequested(&timezoneHours, &timezoneMinutes);

    time_t rawTime = currentTime + timezoneHours * SECONDS_IN_HOUR + timezoneMinutes * SECONDS_IN_MINUTE;

    gmtime_r(&rawTime, &time);

    /* Format artist field */

    char *artist = wavHeader->iart.artist;

    sprintf(artist, "AudioMoth " SERIAL_NUMBER, FORMAT_SERIAL_NUMBER(serialNumber));

    /* Clear comment field */

    char *comment = wavHeader->icmt.comment;

    memset(comment, 0, LENGTH_OF_COMMENT);

    /* Format comment field */

    comment += sprintf(comment, "Recorded at %02d:%02d:%02d %02d/%02d/%04d (UTC", time.tm_hour, time.tm_min, time.tm_sec, time.tm_mday, MONTH_OFFSET + time.tm_mon, YEAR_OFFSET + time.tm_year);

    if (timezoneHours < 0) {

        comment += sprintf(comment, "%ld", timezoneHours);

    } else if (timezoneHours > 0) {

        comment += sprintf(comment, "+%ld", timezoneHours);

    } else {

        if (timezoneMinutes < 0) comment += sprintf(comment, "-%ld", timezoneHours);

        if (timezoneMinutes > 0) comment += sprintf(comment, "+%ld", timezoneHours);

    }

    if (timezoneMinutes < 0) comment += sprintf(comment, ":%02ld", -timezoneMinutes);

    if (timezoneMinutes > 0) comment += sprintf(comment, ":%02ld", timezoneMinutes);

    if (memcmp(deploymentID, defaultDeploymentID, DEPLOYMENT_ID_LENGTH)) {

        comment += sprintf(comment, ") during deployment " SERIAL_NUMBER " ", FORMAT_SERIAL_NUMBER(deploymentID));

    } else {

        comment += sprintf(comment, ") by %s ", artist);

    }

    if (externalMicrophone) {

        comment += sprintf(comment, "using external microphone ");

    }

    static char *gainSettings[5] = {"low", "low-medium", "medium", "medium-high", "high"};

    comment += sprintf(comment, "at %s gain while battery was ", gainSettings[configSettings->gain]);

    if (extendedBatteryState == AM_EXT_BAT_LOW) {

        comment += sprintf(comment, "less than 2.5V");

    } else if (extendedBatteryState >= AM_EXT_BAT_FULL) {

        comment += sprintf(comment, "greater than 4.9V");

    } else {

        uint32_t batteryVoltage =  extendedBatteryState + AM_EXT_BAT_STATE_OFFSET / AM_BATTERY_STATE_INCREMENT;

        comment += sprintf(comment, "%01lu.%01luV", batteryVoltage / 10, batteryVoltage % 10);

    }

    char *temperatureSign = temperature < 0 ? "-" : "";

    uint32_t temperatureInDecidegrees = ROUNDED_DIV(ABS(temperature), 100);

    comment += sprintf(comment, " and temperature was %s%lu.%luC.", temperatureSign, temperatureInDecidegrees / 10, temperatureInDecidegrees % 10);
    
    bool frequencyTriggerEnabled = configSettings->enableFrequencyTrigger;

    bool amplitudeThresholdEnabled = frequencyTriggerEnabled ? false : configSettings->amplitudeThreshold > 0 || configSettings->enableAmplitudeThresholdDecibelScale || configSettings->enableAmplitudeThresholdPercentageScale;

    if (frequencyTriggerEnabled) {

        comment += sprintf(comment, " Frequency trigger (%u.%ukHz and window length of %u samples) threshold was ", configSettings->frequencyTriggerCentreFrequency / 10, configSettings->frequencyTriggerCentreFrequency % 10, (0x01 << configSettings->frequencyTriggerWindowLengthShift));

        comment += formatPercentage(comment, configSettings->frequencyTriggerThresholdPercentageMantissa, configSettings->frequencyTriggerThresholdPercentageExponent);

        comment += sprintf(comment, " with %us minimum trigger duration.", configSettings->minimumTriggerDuration);

    }

    uint16_t lowerFilterFreq = configSettings->lowerFilterFreq;

    uint16_t higherFilterFreq = configSettings->higherFilterFreq;

    if (filterType == LOW_PASS_FILTER) {

        comment += sprintf(comment, " Low-pass filter with frequency of %01u.%01ukHz applied.", higherFilterFreq / 10, higherFilterFreq % 10);

    } else if (filterType == BAND_PASS_FILTER) {

        comment += sprintf(comment, " Band-pass filter with frequencies of %01u.%01ukHz and %01u.%01ukHz applied.", lowerFilterFreq / 10, lowerFilterFreq % 10, higherFilterFreq / 10, higherFilterFreq % 10);

    } else if (filterType == HIGH_PASS_FILTER) {

        comment += sprintf(comment, " High-pass filter with frequency of %01u.%01ukHz applied.", lowerFilterFreq / 10, lowerFilterFreq % 10);

    }

    if (amplitudeThresholdEnabled) {
        
        comment += sprintf(comment, " Amplitude threshold was ");

        if (configSettings->enableAmplitudeThresholdDecibelScale && configSettings->enableAmplitudeThresholdPercentageScale == false) {

            comment += formatDecibels(comment, configSettings->amplitudeThresholdDecibels, true);

        } else if (configSettings->enableAmplitudeThresholdPercentageScale && configSettings->enableAmplitudeThresholdDecibelScale == false) {

            comment += formatPercentage(comment, configSettings->amplitudeThresholdPercentageMantissa, configSettings->amplitudeThresholdPercentageExponent);

        } else {

            comment += sprintf(comment, "%u", configSettings->amplitudeThreshold);

        }

        comment += sprintf(comment, " with %us minimum trigger duration.", configSettings->minimumTriggerDuration);

    }

    if (recordingState != RECORDING_OKAY) {

        comment += sprintf(comment, " Recording stopped");

        if (recordingState == MICROPHONE_CHANGED) {

            comment += sprintf(comment, " due to microphone change.");

        } else if (recordingState == SWITCH_CHANGED) {

            comment += sprintf(comment, " due to switch position change.");

        } else if (recordingState == MAGNETIC_SWITCH) {
        
            comment += sprintf(comment, " by magnetic switch.");

        } else if (recordingState == SUPPLY_VOLTAGE_LOW) {

            comment += sprintf(comment, " due to low voltage.");

        } else if (recordingState == FILE_SIZE_LIMITED) {

            comment += sprintf(comment, " due to file size limit.");

        } else if (recordingState == SDCARD_WRITE_ERROR) {

            comment += sprintf(comment, " due to SD card write error.");

        }

    }

}

/* Function to write the GUANO data */

static uint32_t writeGuanoData(char *buffer, configSettings_t *configSettings, uint32_t currentTime, bool gpsLocationReceived, int32_t *gpsLastFixLatitude, int32_t *gpsLastFixLongitude, bool acousticLocationReceived, int32_t *acousticLatitude, int32_t *acousticLongitude, uint8_t *firmwareDescription, uint8_t *firmwareVersion, uint8_t *serialNumber, uint8_t *deploymentID, uint8_t *defaultDeploymentID, char *filename, AM_extendedBatteryState_t extendedBatteryState, int32_t temperature, AM_filterType_t filterType) {

    uint32_t length = sprintf(buffer, "guan") + UINT32_SIZE_IN_BYTES;

    /* General information */
    
    length += sprintf(buffer + length, "GUANO|Version:1.0\nMake:Open Acoustic Devices\nModel:AudioMoth\nSerial:" SERIAL_NUMBER "\n", FORMAT_SERIAL_NUMBER(serialNumber));

    if (memcmp(deploymentID, defaultDeploymentID, DEPLOYMENT_ID_LENGTH)) {

        length += sprintf(buffer + length, "OAD|Deployment ID:" SERIAL_NUMBER "\n", FORMAT_SERIAL_NUMBER(deploymentID));

    }

    length += sprintf(buffer + length, "Firmware Version:%s (%u.%u.%u)\n", firmwareDescription, firmwareVersion[0], firmwareVersion[1], firmwareVersion[2]);

    /* Timestamp */

    int32_t timezoneHours, timezoneMinutes;

    AudioMoth_timezoneRequested(&timezoneHours, &timezoneMinutes);

    int32_t timezoneSeconds = timezoneHours * SECONDS_IN_HOUR + timezoneMinutes * SECONDS_IN_MINUTE;

    time_t rawTime = currentTime + timezoneSeconds;

    struct tm time;

    gmtime_r(&rawTime, &time);

    length += sprintf(buffer + length, "Timestamp:%04d-%02d-%02dT%02d:%02d:%02d", YEAR_OFFSET + time.tm_year, MONTH_OFFSET + time.tm_mon, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec);

    if (timezoneSeconds == 0) {

        length += sprintf(buffer + length, "Z\n");
        
    } else if (timezoneSeconds < 0) {

        length += sprintf(buffer + length, "-%02ld:%02ld\n", ABS(timezoneHours), ABS(timezoneMinutes));

    } else {

        length += sprintf(buffer + length, "+%02ld:%02ld\n", timezoneHours, timezoneMinutes);

    }

    /* Location position and source */

    if (gpsLocationReceived || acousticLocationReceived || configSettings->enableSunRecording) {

        int32_t latitude = gpsLocationReceived ? *gpsLastFixLatitude : acousticLocationReceived ? *acousticLatitude : configSettings->latitude;

        int32_t longitude = gpsLocationReceived ? *gpsLastFixLongitude : acousticLocationReceived ? *acousticLongitude : configSettings->longitude;

        char *latitudeSign = latitude < 0 ? "-" : "";

        char *longitudeSign = longitude < 0 ? "-" : "";

        if (gpsLocationReceived) {

            length += sprintf(buffer + length, "Loc Position:%s%ld.%06ld %s%ld.%06ld\nOAD|Loc Source:GPS\n", latitudeSign, ABS(latitude) / GPS_LOCATION_PRECISION, ABS(latitude) % GPS_LOCATION_PRECISION, longitudeSign, ABS(longitude) / GPS_LOCATION_PRECISION, ABS(longitude) % GPS_LOCATION_PRECISION);

        } else if (acousticLocationReceived) {

            length += sprintf(buffer + length, "Loc Position:%s%ld.%06ld %s%ld.%06ld\nOAD|Loc Source:Acoustic chime\n", latitudeSign, ABS(latitude) / ACOUSTIC_LOCATION_PRECISION, ABS(latitude) % ACOUSTIC_LOCATION_PRECISION, longitudeSign, ABS(longitude) / ACOUSTIC_LOCATION_PRECISION, ABS(longitude) % ACOUSTIC_LOCATION_PRECISION);

        } else {

            length += sprintf(buffer + length, "Loc Position:%s%ld.%02ld %s%ld.%02ld\nOAD|Loc Source:Configuration app\n", latitudeSign, ABS(latitude) / CONFIG_LOCATION_PRECISION, ABS(latitude) % CONFIG_LOCATION_PRECISION, longitudeSign, ABS(longitude) / CONFIG_LOCATION_PRECISION, ABS(longitude) % CONFIG_LOCATION_PRECISION);

        }

    }

    /* Filename */

    char *start = strchr(filename, '/');

    length += sprintf(buffer + length, "Original Filename:%s\n", start ? start + 1 : filename);

    /* Recording settings */

    length += sprintf(buffer + length, "OAD|Recording Settings:%lu GAIN %u", configSettings->sampleRate / configSettings->sampleRateDivider, configSettings->gain);

    bool frequencyTriggerEnabled = configSettings->enableFrequencyTrigger;

    bool amplitudeThresholdEnabled = frequencyTriggerEnabled ? false : configSettings->amplitudeThreshold > 0 || configSettings->enableAmplitudeThresholdDecibelScale || configSettings->enableAmplitudeThresholdPercentageScale;

    if (frequencyTriggerEnabled) {

        length += sprintf(buffer + length, " FREQ %u %u ", FILTER_FREQ_MULTIPLIER * configSettings->frequencyTriggerCentreFrequency, 0x01 << configSettings->frequencyTriggerWindowLengthShift);

        length += formatPercentage(buffer + length, configSettings->frequencyTriggerThresholdPercentageMantissa, configSettings->frequencyTriggerThresholdPercentageExponent);

        length += sprintf(buffer + length, " %u", configSettings->minimumTriggerDuration);

    }

    uint32_t lowerFilterFreq = FILTER_FREQ_MULTIPLIER * configSettings->lowerFilterFreq;

    uint32_t higherFilterFreq = FILTER_FREQ_MULTIPLIER * configSettings->higherFilterFreq;

    if (filterType == LOW_PASS_FILTER) {

        length += sprintf(buffer + length, " LPF %lu", higherFilterFreq);

    } else if (filterType == BAND_PASS_FILTER) {

        length += sprintf(buffer + length, " BPF %lu %lu", lowerFilterFreq, higherFilterFreq);

    } else if (filterType == HIGH_PASS_FILTER) {

        length += sprintf(buffer + length, " HPF %lu", lowerFilterFreq);

    }

    if (amplitudeThresholdEnabled) {

        length += sprintf(buffer + length, " AMP ");

        if (configSettings->enableAmplitudeThresholdDecibelScale && configSettings->enableAmplitudeThresholdPercentageScale == false) {

            length += formatDecibels(buffer + length, configSettings->amplitudeThresholdDecibels, false);

        } else if (configSettings->enableAmplitudeThresholdPercentageScale && configSettings->enableAmplitudeThresholdDecibelScale == false) {

            length += formatPercentage(buffer + length, configSettings->amplitudeThresholdPercentageMantissa, configSettings->amplitudeThresholdPercentageExponent);

        } else {

            length += sprintf(buffer + length, "%u", configSettings->amplitudeThreshold);

        }
        
        length += sprintf(buffer + length, " %u", configSettings->minimumTriggerDuration);

    }

    if (configSettings->enableLowGainRange) length += sprintf(buffer + length, " LGR");

    if (configSettings->disable48HzDCBlockingFilter) length += sprintf(buffer + length, " D48");

    if (isEnergySaverMode(configSettings)) length += sprintf(buffer + length, " ESM");

    /* Battery and temperature */

    uint32_t batteryVoltage = extendedBatteryState == AM_EXT_BAT_LOW ? 24 : extendedBatteryState >= AM_EXT_BAT_FULL ? 50 : extendedBatteryState + AM_EXT_BAT_STATE_OFFSET / AM_BATTERY_STATE_INCREMENT;

    length += sprintf(buffer + length, "\nOAD|Battery Voltage:%01lu.%01lu\n", batteryVoltage / 10, batteryVoltage % 10);
    
    char *temperatureSign = temperature < 0 ? "-" : "";

    uint32_t temperatureInDecidegrees = ROUNDED_DIV(ABS(temperature), 100);

    length += sprintf(buffer + length, "Temperature Int:%s%lu.%lu", temperatureSign, temperatureInDecidegrees / 10, temperatureInDecidegrees % 10);

    /* Set GUANO chunk size */

    *(uint32_t*)(buffer + RIFF_ID_LENGTH) = length - sizeof(chunk_t);;

    return length;

}

/* Function to write configuration to file */

static bool writeConfigurationToFile(char *buffer, configSettings_t *configSettings, uint32_t currentTime, bool gpsLocationReceived, int32_t *gpsLatitude, int32_t *gpsLongitude, bool acousticLocationReceived, int32_t *acousticLatitude, int32_t *acousticLongitude, uint8_t *firmwareDescription, uint8_t *firmwareVersion, uint8_t *serialNumber, uint8_t *deploymentID, uint8_t *defaultDeploymentID) {

    static char timezoneBuffer[CONFIG_TIMEZONE_LENGTH];

    int32_t timezoneHours, timezoneMinutes;

    AudioMoth_timezoneRequested(&timezoneHours, &timezoneMinutes);

    int32_t timezoneSeconds = timezoneHours * SECONDS_IN_HOUR + timezoneMinutes * SECONDS_IN_MINUTE;

    RETURN_BOOL_ON_ERROR(AudioMoth_openFile("CONFIG.TXT"));

    uint32_t length = sprintf(buffer, "Device ID                       : " SERIAL_NUMBER "\r\n", FORMAT_SERIAL_NUMBER(serialNumber));

    length += sprintf(buffer + length, "Firmware                        : %s (%u.%u.%u)\r\n\r\n", firmwareDescription, firmwareVersion[0], firmwareVersion[1], firmwareVersion[2]);

    if (memcmp(deploymentID, defaultDeploymentID, DEPLOYMENT_ID_LENGTH)) {

        length += sprintf(buffer + length, "Deployment ID                   : " SERIAL_NUMBER "\r\n\r\n", FORMAT_SERIAL_NUMBER(deploymentID));

    }

    uint32_t timezoneLength = sprintf(timezoneBuffer, "UTC");

    if (timezoneHours < 0) {

        timezoneLength += sprintf(timezoneBuffer + timezoneLength, "%ld", timezoneHours);

    } else if (timezoneHours > 0) {

        timezoneLength += sprintf(timezoneBuffer + timezoneLength, "+%ld", timezoneHours);

    } else {

        if (timezoneMinutes < 0) timezoneLength += sprintf(timezoneBuffer + timezoneLength, "-%ld", timezoneHours);

        if (timezoneMinutes > 0) timezoneLength += sprintf(timezoneBuffer + timezoneLength, "+%ld", timezoneHours);

    }

    if (timezoneMinutes < 0) timezoneLength += sprintf(timezoneBuffer + timezoneLength, ":%02ld", -timezoneMinutes);

    if (timezoneMinutes > 0) timezoneLength += sprintf(timezoneBuffer + timezoneLength, ":%02ld", timezoneMinutes);

    time_t rawTime = currentTime + timezoneSeconds;

    struct tm time;

    gmtime_r(&rawTime, &time);

    length += sprintf(buffer + length, "Device time                     : %04d-%02d-%02d %02d:%02d:%02d (%s)", YEAR_OFFSET + time.tm_year, MONTH_OFFSET + time.tm_mon, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec, timezoneBuffer);

    RETURN_BOOL_ON_ERROR(AudioMoth_writeToFile(buffer, length));

    length = sprintf(buffer, "\r\n\r\nSample rate (Hz)                : %lu\r\n", configSettings->sampleRate / configSettings->sampleRateDivider);

    static char *gainSettings[5] = {"Low", "Low-Medium", "Medium", "Medium-High", "High"};

    length += sprintf(buffer + length, "Gain                            : %s\r\n\r\n", gainSettings[configSettings->gain]);

    length += sprintf(buffer + length, "Sleep duration (s)              : ");

    if (configSettings->disableSleepRecordCycle) {

        length += sprintf(buffer + length, "-");

    } else {

        length += sprintf(buffer + length, "%u", configSettings->sleepDuration);

    }

    length += sprintf(buffer + length, "\r\nRecording duration (s)          : ");

    if (configSettings->disableSleepRecordCycle) {

        length += sprintf(buffer + length, "-");

    } else {

        length += sprintf(buffer + length, "%u", configSettings->recordDuration);

    }

    RETURN_BOOL_ON_ERROR(AudioMoth_writeToFile(buffer, length));

    if (configSettings->enableSunRecording) {

        int32_t latitude = gpsLocationReceived ? *gpsLatitude : acousticLocationReceived ? *acousticLatitude : configSettings->latitude;

        int32_t longitude = gpsLocationReceived ? *gpsLongitude : acousticLocationReceived ? *acousticLongitude : configSettings->longitude;

        char latitudeDirection = latitude < 0 ? 'S' : 'N';

        char longitudeDirection = longitude < 0 ? 'W' : 'E';

        if (gpsLocationReceived) {

            length = sprintf(buffer, "\r\n\r\nLocation                        : %ld.%06ld°%c %ld.%06ld°%c (GPS)", ABS(latitude) / GPS_LOCATION_PRECISION, ABS(latitude) % GPS_LOCATION_PRECISION, latitudeDirection, ABS(longitude) / GPS_LOCATION_PRECISION, ABS(longitude) % GPS_LOCATION_PRECISION, longitudeDirection);

        } else if (acousticLocationReceived) {

            length = sprintf(buffer, "\r\n\r\nLocation                        : %ld.%06ld°%c %ld.%06ld°%c (Acoustic chime)", ABS(latitude) / ACOUSTIC_LOCATION_PRECISION, ABS(latitude) % ACOUSTIC_LOCATION_PRECISION, latitudeDirection, ABS(longitude) / ACOUSTIC_LOCATION_PRECISION, ABS(longitude) % ACOUSTIC_LOCATION_PRECISION, longitudeDirection);

        } else {

            length = sprintf(buffer, "\r\n\r\nLocation                        : %ld.%02ld°%c %ld.%02ld°%c (Configuration app)", ABS(latitude) / CONFIG_LOCATION_PRECISION, ABS(latitude) % CONFIG_LOCATION_PRECISION, latitudeDirection, ABS(longitude) / CONFIG_LOCATION_PRECISION, ABS(longitude) % CONFIG_LOCATION_PRECISION, longitudeDirection);

        }

        static char* twilightTypes[3] = {"Civil", "Nautical", "Astronomical"};

        static char* dawnDuskModes[5] = {"dawn", "dusk", "dawn and dusk", "dusk to dawn", "dawn to dusk"};

        static char* sunriseSunsetModes[5] = {"Sunrise", "Sunset", "Sunrise and sunset", "Sunset to sunrise", "Sunrise to sunset"};

        length += sprintf(buffer + length, "\r\nSun recording mode              : ");
        
        if (configSettings->sunRecordingEvent == SR_SUNRISE_AND_SUNSET) {

            length += sprintf(buffer + length, "%s", sunriseSunsetModes[configSettings->sunRecordingMode]);

        } else {

            length += sprintf(buffer + length, "%s %s", twilightTypes[configSettings->sunRecordingEvent - 1], dawnDuskModes[configSettings->sunRecordingMode]);

        }

        char *sunriseText = configSettings->sunRecordingEvent == SR_SUNRISE_AND_SUNSET ? "\r\nSunrise - before, after (mins)  " : "\r\nDawn - before, after (mins)     ";

        char *sunsetText = configSettings->sunRecordingEvent == SR_SUNRISE_AND_SUNSET ? "\r\nSunset - before, after (mins)   " : "\r\nDusk - before, after (mins)     ";

        if (configSettings->sunRecordingMode == SUNRISE_RECORDING) {

            length += sprintf(buffer + length, "%s: %u, %u", sunriseText, configSettings->beforeSunriseMinutes, configSettings->afterSunriseMinutes);
            length += sprintf(buffer + length, "%s: -, -", sunsetText);
            
        } else if (configSettings->sunRecordingMode == SUNSET_RECORDING) {

            length += sprintf(buffer + length, "%s: -, -", sunriseText);
            length += sprintf(buffer + length, "%s: %u, %u", sunsetText, configSettings->beforeSunsetMinutes, configSettings->afterSunsetMinutes);

        } else if (configSettings->sunRecordingMode == SUNRISE_AND_SUNSET_RECORDING) {

            length += sprintf(buffer + length, "%s: %u, %u", sunriseText, configSettings->beforeSunriseMinutes, configSettings->afterSunriseMinutes);
            length += sprintf(buffer + length, "%s: %u, %u", sunsetText, configSettings->beforeSunsetMinutes, configSettings->afterSunsetMinutes);

        } else if (configSettings->sunRecordingMode == SUNSET_TO_SUNRISE_RECORDING) {

            length += sprintf(buffer + length, "%s: -, %u", sunriseText, configSettings->afterSunriseMinutes);
            length += sprintf(buffer + length, "%s: %u, -", sunsetText, configSettings->beforeSunsetMinutes);

        } else if (configSettings->sunRecordingMode == SUNRISE_TO_SUNSET_RECORDING) {

            length += sprintf(buffer + length, "%s: %u, -", sunriseText, configSettings->beforeSunriseMinutes);
            length += sprintf(buffer + length, "%s: -, %u", sunsetText, configSettings->afterSunsetMinutes);

        }

        char *roundingText = configSettings->sunRecordingEvent == SR_SUNRISE_AND_SUNSET ? "\r\nSunrise/sunset rounding (mins)  : %u" : "\r\nDawn/dusk rounding (mins)       : %u";

        length += sprintf(buffer + length, roundingText, configSettings->sunRoundingMinutes);

    } else {

        length = sprintf(buffer, "\r\n\r\nActive recording periods        : %u\r\n", configSettings->activeRecordingPeriods);

        /* Calculate the start time offset for the appropriate time zone */

        int32_t startMinutesOffset = timezoneSeconds / SECONDS_IN_MINUTE;

        if (configSettings->useTimezoneFromAcousticChime && configSettings->adjustScheduleUsingTimezoneFromAcousticChime) {

            startMinutesOffset = configSettings->timezoneHours * MINUTES_IN_HOUR + configSettings->timezoneMinutes;

        }

        /* Find the first recording period */

        uint32_t minimumIndex = 0;

        uint32_t minimumStartMinutes = UINT32_MAX;

        for (uint32_t i = 0; i < configSettings->activeRecordingPeriods; i += 1) {

            uint32_t startMinutes = (MINUTES_IN_DAY + configSettings->recordingPeriods[i].startMinutes + startMinutesOffset) % MINUTES_IN_DAY;

            if (startMinutes < minimumStartMinutes) {

                minimumStartMinutes = startMinutes;

                minimumIndex = i;

            }

        }
    
        /* Display the recording periods */

        for (uint32_t i = 0; i < configSettings->activeRecordingPeriods; i += 1) {

            uint32_t index = (minimumIndex + i) % configSettings->activeRecordingPeriods;

            uint32_t startMinutes = (MINUTES_IN_DAY + configSettings->recordingPeriods[index].startMinutes + startMinutesOffset) % MINUTES_IN_DAY;

            uint32_t endMinutes = (MINUTES_IN_DAY + configSettings->recordingPeriods[index].endMinutes + startMinutesOffset) % MINUTES_IN_DAY;

            length += sprintf(buffer + length, "\r\nRecording period %lu              : %02lu:%02lu - %02lu:%02lu (%s)", i + 1, startMinutes / MINUTES_IN_HOUR, startMinutes % MINUTES_IN_HOUR, endMinutes / MINUTES_IN_HOUR, endMinutes % MINUTES_IN_HOUR, timezoneBuffer);

        }

    }

    RETURN_BOOL_ON_ERROR(AudioMoth_writeToFile(buffer, length));

    if (configSettings->earliestRecordingTime == 0) {

        length = sprintf(buffer, "\r\n\r\nFirst recording date            : ----------");

    } else {

        time_t rawTime = configSettings->earliestRecordingTime + timezoneSeconds;

        if (configSettings->useTimezoneFromAcousticChime && configSettings->adjustScheduleUsingTimezoneFromAcousticChime) {

            rawTime += configSettings->timezoneHours * SECONDS_IN_HOUR + configSettings->timezoneMinutes * SECONDS_IN_MINUTE - timezoneSeconds;

        }

        gmtime_r(&rawTime, &time);

        if (time.tm_hour == 0 && time.tm_min == 0 && time.tm_sec == 0) {

            length = sprintf(buffer, "\r\n\r\nFirst recording date            : ");

            length += sprintf(buffer + length, "%04d-%02d-%02d (%s)", YEAR_OFFSET + time.tm_year, MONTH_OFFSET + time.tm_mon, time.tm_mday, timezoneBuffer);

        } else {

            length = sprintf(buffer, "\r\n\r\nFirst recording time            : ");

            length += sprintf(buffer + length, "%04d-%02d-%02d %02d:%02d:%02d (%s)", YEAR_OFFSET + time.tm_year, MONTH_OFFSET + time.tm_mon, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec, timezoneBuffer);

        } 

    }
                                              
    if (configSettings->latestRecordingTime == 0) {

        length += sprintf(buffer + length, "\r\nLast recording date             : ----------");

    } else {

        time_t rawTime = configSettings->latestRecordingTime + timezoneSeconds;

        if (configSettings->useTimezoneFromAcousticChime && configSettings->adjustScheduleUsingTimezoneFromAcousticChime) {

            rawTime += configSettings->timezoneHours * SECONDS_IN_HOUR + configSettings->timezoneMinutes * SECONDS_IN_MINUTE - timezoneSeconds;

        }

        gmtime_r(&rawTime, &time);

        if (time.tm_hour == 0 && time.tm_min == 0 && time.tm_sec == 0) {

            rawTime -= SECONDS_IN_DAY;

            gmtime_r(&rawTime, &time);

            length += sprintf(buffer + length, "\r\nLast recording date             : ");

            length += sprintf(buffer + length, "%04d-%02d-%02d (%s)", YEAR_OFFSET + time.tm_year, MONTH_OFFSET + time.tm_mon, time.tm_mday, timezoneBuffer);

        } else {

            length += sprintf(buffer + length, "\r\nLast recording time             : ");

            length += sprintf(buffer + length, "%04d-%02d-%02d %02d:%02d:%02d (%s)", YEAR_OFFSET + time.tm_year, MONTH_OFFSET + time.tm_mon, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec, timezoneBuffer);

        }

    }

    RETURN_BOOL_ON_ERROR(AudioMoth_writeToFile(buffer, length));

    length = sprintf(buffer, "\r\n\r\nFilter                          : ");

    if (configSettings->lowerFilterFreq == 0 && configSettings->higherFilterFreq == 0) {

        length += sprintf(buffer + length, "-");

    } else if (configSettings->lowerFilterFreq == UINT16_MAX) {

        length += sprintf(buffer + length, "Low-pass (%u.%ukHz)", configSettings->higherFilterFreq / 10, configSettings->higherFilterFreq % 10);

    } else if (configSettings->higherFilterFreq == UINT16_MAX) {

        length += sprintf(buffer + length, "High-pass (%u.%ukHz)", configSettings->lowerFilterFreq / 10, configSettings->lowerFilterFreq % 10);

    } else {

        length += sprintf(buffer + length, "Band-pass (%u.%ukHz - %u.%ukHz)", configSettings->lowerFilterFreq / 10, configSettings->lowerFilterFreq % 10, configSettings->higherFilterFreq / 10, configSettings->higherFilterFreq % 10);

    }

    bool frequencyTriggerEnabled = configSettings->enableFrequencyTrigger;

    bool amplitudeThresholdEnabled = frequencyTriggerEnabled ? false : configSettings->amplitudeThreshold > 0 || configSettings->enableAmplitudeThresholdDecibelScale || configSettings->enableAmplitudeThresholdPercentageScale;

    length += sprintf(buffer + length, "\r\n\r\nTrigger type                    : ");

    if (frequencyTriggerEnabled) {

        length += sprintf(buffer + length, "Frequency (%u.%ukHz and window length of %u samples)", configSettings->frequencyTriggerCentreFrequency / 10, configSettings->frequencyTriggerCentreFrequency % 10, (0x01 << configSettings->frequencyTriggerWindowLengthShift));

        length += sprintf(buffer + length, "\r\nThreshold setting               : ");

        length += formatPercentage(buffer + length, configSettings->frequencyTriggerThresholdPercentageMantissa, configSettings->frequencyTriggerThresholdPercentageExponent);

    } else if (amplitudeThresholdEnabled) {

        length += sprintf(buffer + length, "Amplitude");

        length += sprintf(buffer + length, "\r\nThreshold setting               : ");

        if (configSettings->enableAmplitudeThresholdDecibelScale && configSettings->enableAmplitudeThresholdPercentageScale == false) {

            length += formatDecibels(buffer + length, configSettings->amplitudeThresholdDecibels, true);

        } else if (configSettings->enableAmplitudeThresholdPercentageScale && configSettings->enableAmplitudeThresholdDecibelScale == false) {

            length += formatPercentage(buffer + length, configSettings->amplitudeThresholdPercentageMantissa, configSettings->amplitudeThresholdPercentageExponent);

        } else {

            length += sprintf(buffer + length, "%u", configSettings->amplitudeThreshold);

        }

    } else {

        length += sprintf(buffer + length, "-");

        length += sprintf(buffer + length, "\r\nThreshold setting               : -");

    }

    length += sprintf(buffer + length, "\r\nMinimum trigger duration (s)    : ");

    if (frequencyTriggerEnabled || amplitudeThresholdEnabled) {

        length += sprintf(buffer + length, "%u", configSettings->minimumTriggerDuration);

    } else {

        length += sprintf(buffer + length, "-");

    }

    RETURN_BOOL_ON_ERROR(AudioMoth_writeToFile(buffer, length));

    length = sprintf(buffer, "\r\n\r\nEnable LED                      : %s\r\n", configSettings->enableLED ? "Yes" : "No");

    length += sprintf(buffer + length, "Enable low-voltage cut-off      : %s\r\n", configSettings->enableLowVoltageCutoff ? "Yes" : "No");

    length += sprintf(buffer + length, "Enable battery level indication : %s\r\n\r\n", configSettings->disableBatteryLevelDisplay ? "No" : configSettings->batteryLevelDisplayType == NIMH_LIPO_BATTERY_VOLTAGE ? "Yes (NiMH/LiPo voltage range)" : "Yes");

    length += sprintf(buffer + length, "Always require acoustic chime   : %s\r\n", configSettings->requireAcousticConfiguration ? "Yes" : "No");
    
    length += sprintf(buffer + length, "Also require location in chime  : %s\r\n", configSettings->requireAcousticConfiguration == false ? "-" : configSettings->requireAcousticLocation ? "Yes" : "No");

    length += sprintf(buffer + length, "Use timezone from chime         : %s\r\n", configSettings->useTimezoneFromAcousticChime ? "Yes" : "No");

    length += sprintf(buffer + length, "Also adjust recording schedule  : %s\r\n\r\n", configSettings->useTimezoneFromAcousticChime == false ? "-" : configSettings->adjustScheduleUsingTimezoneFromAcousticChime ? "Yes" : "No");

    RETURN_BOOL_ON_ERROR(AudioMoth_writeToFile(buffer, length));

    length = sprintf(buffer, "Recording preparation time (s)  : %d\r\n", configSettings->preparationPeriodBuffer == 0 ? DEFAULT_PREPARATION_PERIOD_BUFFER / MILLISECONDS_IN_SECOND : configSettings->preparationPeriodBuffer);

    length += sprintf(buffer + length, "Use device ID in WAV file name  : %s\r\n", configSettings->enableFilenameWithDeviceID ? "Yes" : "No");

    length += sprintf(buffer + length, "Use daily folder for WAV files  : %s\r\n\r\n", configSettings->enableDailyFolders ? "Yes" : "No");

    length += sprintf(buffer + length, "Disable 48Hz DC blocking filter : %s\r\n", configSettings->disable48HzDCBlockingFilter ? "Yes" : "No");

    length += sprintf(buffer + length, "Enable energy saver mode        : %s\r\n", configSettings->enableEnergySaverMode ? "Yes" : "No");

    length += sprintf(buffer + length, "Enable low gain range           : %s\r\n\r\n", configSettings->enableLowGainRange ? "Yes" : "No");

    RETURN_BOOL_ON_ERROR(AudioMoth_writeToFile(buffer, length));

    length = sprintf(buffer, "Ignore external microphone      : %s\r\n\r\n", configSettings->ignoreExternalMicrophoneForAcousticChime ? "Yes" : "No");

    length += sprintf(buffer + length, "Enable magnetic switch          : %s\r\n\r\n", configSettings->enableMagneticSwitch ? "Yes" : "No");

    RETURN_BOOL_ON_ERROR(AudioMoth_writeToFile(buffer, length));

    length = sprintf(buffer, "Enable GPS time setting         : %s\r\n", configSettings->enableTimeSettingFromGPS ? "Yes" : "No");

    length += sprintf(buffer + length, "GPS fix before and after        : %s\r\n", configSettings->enableTimeSettingFromGPS == false ? "-" : configSettings->enableTimeSettingBeforeAndAfterRecordings ? "Individual recordings" : "Recording periods");

    length += sprintf(buffer + length, "GPS fix time (mins)             : ");

    if (configSettings->enableTimeSettingFromGPS) {

        uint32_t gpsTimeSettingPeriod = configSettings->gpsTimeSettingPeriod == 0 ? GPS_DEFAULT_TIME_SETTING_PERIOD / SECONDS_IN_MINUTE : configSettings->gpsTimeSettingPeriod;

        length += sprintf(buffer + length, "%ld\r\n", gpsTimeSettingPeriod);

    } else {

        length += sprintf(buffer + length, "-\r\n");

    }

    RETURN_BOOL_ON_ERROR(AudioMoth_writeToFile(buffer, length));

    RETURN_BOOL_ON_ERROR(AudioMoth_closeFile());

    return true;

}

/* Backup domain variables */

static uint32_t *backupDomainFlags = (uint32_t*)AM_BACKUP_DOMAIN_START_ADDRESS;

static uint32_t *previousSwitchPosition = (uint32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 4);

static uint32_t *startOfRecordingPeriod = (uint32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 8);

static uint32_t *timeOfNextRecording = (uint32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 12);

static uint32_t *indexOfNextRecording = (uint32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 16);

static uint32_t *durationOfNextRecording = (uint32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 20);

static uint32_t *timeOfNextGPSTimeSetting = (uint32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 24);

static uint8_t *deploymentID = (uint8_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 28);

static uint32_t *numberOfRecordingErrors = (uint32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 36);

static uint32_t *numberOfConsecutiveRecordingErrors = (uint32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 40);

static uint32_t *recordingPreparationPeriod = (uint32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 44);

static int32_t *gpsLatitude = (int32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 48);

static int32_t *gpsLongitude = (int32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 52);

static int32_t *gpsLastFixLatitude = (int32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 56);

static int32_t *gpsLastFixLongitude = (int32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 60);

static int32_t *acousticLatitude = (int32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 64);

static int32_t *acousticLongitude = (int32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 68);

static int32_t *acousticTimezoneMinutes = (int32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 72);

static uint32_t *numberOfSunRecordingPeriods = (uint32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 76);

static recordingPeriod_t *firstSunRecordingPeriod = (recordingPeriod_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 80);

static recordingPeriod_t *secondSunRecordingPeriod = (recordingPeriod_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 84);

static uint32_t *timeOfNextSunriseSunsetCalculation = (uint32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 88);

static uint32_t *timeAtWhichToSwitchOffLED = (uint32_t*)(AM_BACKUP_DOMAIN_START_ADDRESS + 92);

/* Functions to query, set and clear backup domain flags */

typedef enum {
    BACKUP_WRITTEN_CONFIGURATION_TO_FILE,
    BACKUP_READY_TO_MAKE_RECORDING, 
    BACKUP_MUST_SET_TIME_FROM_GPS, 
    BACKUP_SHOULD_SET_TIME_FROM_GPS, 
    BACKUP_WAITING_FOR_MAGNETIC_SWITCH,
    BACKUP_POWERED_DOWN_WITH_SHORT_WAIT_INTERVAL,
    BACKUP_GPS_LOCATION_RECEIVED,
    BACKUP_ACOUSTIC_LOCATION_RECEIVED,
    BACKUP_ACOUSTIC_TIMEZONE_RECEIVED
} AM_backupDomainFlag_t;

static inline bool getBackupFlag(AM_backupDomainFlag_t flag) {
    uint32_t mask = (1 << flag);
    return (*backupDomainFlags & mask) == mask;
}

static inline void setBackupFlag(AM_backupDomainFlag_t flag, bool state) {
    if (state) {
        *backupDomainFlags |= (1 << flag);
    } else {
        *backupDomainFlags &= ~(1 << flag);
    }
}

/* Filter variables */

static AM_filterType_t requestedFilterType;

/* DMA transfer variable */

static uint32_t numberOfRawSamplesInDMATransfer;

/* SRAM buffer variables */

static volatile uint32_t writeBuffer;

static volatile uint32_t writeBufferIndex;

static int16_t* buffers[NUMBER_OF_BUFFERS];

/* Flag to start processing DMA transfers */

static volatile uint32_t numberOfDMATransfers;

static volatile uint32_t numberOfDMATransfersToWait;

/* Write indicator buffer */

static bool writeIndicator[NUMBER_OF_BUFFERS];

/* GPS message buffer */

static char gpsMessageBuffer[GPS_MESSAGE_BUFFER_SIZE_IN_BYTES];

/* Compression buffer */

static int16_t compressionBuffer[COMPRESSION_BUFFER_SIZE_IN_BYTES / NUMBER_OF_BYTES_IN_SAMPLE] __attribute__ ((aligned(UINT32_SIZE_IN_BYTES)));

/* Configuration settings */

static configSettings_t *configSettings = &((persistentConfigSettings_t*)AM_FLASH_USER_DATA_ADDRESS)->configSettings;

static persistentConfigSettings_t *persistentConfigSettings = (persistentConfigSettings_t*)AM_FLASH_USER_DATA_ADDRESS;

static persistentConfigSettings_t *tempPersistentConfigSettings = (persistentConfigSettings_t*)compressionBuffer;

/* LED variable */

static bool enableLED;

/* GPS fix variables */

static bool gpsEnableLED;

static bool gpsPPSEvent;

static bool gpsFixEvent;

static bool gpsMessageEvent;

static uint32_t timeSetFromGPS;

static bool gpsFirstMessageReceived;

static uint32_t gpsTickEventCount = 1;

static uint32_t gpsTickEventModulo = GPS_TICK_EVENTS_PER_SECOND;

/* Audio configuration variables */

static bool audioConfigStateLED;

static bool audioConfigToggleLED;

static uint32_t audioConfigPulseCounter;

static bool acousticConfigurationPerformed;

static uint32_t secondsOfAcousticSignalStart;

static uint32_t millisecondsOfAcousticSignalStart;

/* Deployment ID variable */

static uint8_t defaultDeploymentID[DEPLOYMENT_ID_LENGTH];

/* Recording state */

static volatile bool magneticSwitch;

static volatile bool microphoneChanged;

static volatile bool switchPositionChanged;

/* DMA buffers */

static int16_t primaryBuffer[MAXIMUM_SAMPLES_IN_DMA_TRANSFER];

static int16_t secondaryBuffer[MAXIMUM_SAMPLES_IN_DMA_TRANSFER];

/* Firmware version and description */

static uint8_t firmwareVersion[AM_FIRMWARE_VERSION_LENGTH] = {1, 12, 0};

static uint8_t firmwareDescription[AM_FIRMWARE_DESCRIPTION_LENGTH] = "AudioMoth-Firmware-Basic";

/* Function prototypes */

static AM_recordingState_t makeRecording(uint32_t timeOfNextRecording, uint32_t recordDuration, uint32_t timeAtWhichToSwitchOffLED, AM_extendedBatteryState_t extendedBatteryState, int32_t temperature, uint32_t *fileOpenTime, uint32_t *fileOpenMilliseconds);

#if STFT_DEBUG_MODE
static void runSTFTFeasibilityTest(void);
#endif

#if TRIGGERED_RECORDING_MODE
static void runTriggeredRecording(void);
#endif

/* Forward-declared so TRIGGERED_RECORDING_MODE's openTriggeredWavFile() (defined well
   before makeRecording() further down, where this is otherwise first defined) can call it -
   pure/stateless (a timestamp + bool in, a folder/filename out), safe to call from a
   wholly different code path. */

static void generateFolderAndFilename(char *foldername, char *filename, uint32_t timestamp, bool triggeredRecording);

static void scheduleRecording(uint32_t currentTime, uint32_t *timeOfNextRecording, uint32_t *indexOfNextRecording, uint32_t *durationOfNextRecording, uint32_t *startOfRecordingPeriod, uint32_t *endOfRecordingPeriod);

static void determineTimeOfNextSunriseSunsetCalculation(uint32_t currentTime, uint32_t *timeOfNextSunriseSunsetCalculation);

static void determineSunriseAndSunsetTimesAndScheduleRecording(uint32_t currentTime);

static void determineSunriseAndSunsetTimes(uint32_t currentTime);

static void flashLedToIndicateBatteryLife(void);

/* Functions of copy to and from the backup domain */

static void copyToBackupDomain(uint32_t *dst, uint8_t *src, uint32_t length) {

    uint32_t value = 0;

    for (uint32_t i = 0; i < length / UINT32_SIZE_IN_BYTES; i += 1) {
        *(dst + i) = *((uint32_t*)src + i);
    }

    for (uint32_t i = 0; i < length % UINT32_SIZE_IN_BYTES; i += 1) {
        value = (value << BITS_PER_BYTE) + *(src + length - 1 - i);
    }

    if (length % UINT32_SIZE_IN_BYTES) *(dst + length / UINT32_SIZE_IN_BYTES) = value;

}

/* Seeds deploymentID with its all-zero default. Only ever meaningful to call once per cold
   boot (inside an `if (AudioMoth_isInitialPowerUp())` guard) - factored out so both the
   normal power-up path below AND runTriggeredRecording() (which, on a genuine cold boot
   straight into that mode, never reaches the normal power-up block at all - see main())
   can seed it, each behind their own single AudioMoth_isInitialPowerUp() check, without
   calling that function twice in the same run. Without this, generateFolderAndFilename()
   would embed electrically-arbitrary backup-RAM content in every triggered filename
   whenever enableFilenameWithDeviceID is set. */

static void initialiseDeploymentID(void) {

    copyToBackupDomain((uint32_t*)deploymentID, (uint8_t*)defaultDeploymentID, DEPLOYMENT_ID_LENGTH);

}

/* GPS time setting functions */

static void writeGPSLogMessage(char *buffer, uint32_t currentTime, uint32_t currentMilliseconds, char *message) {

    struct tm time;

    time_t rawTime = currentTime;

    gmtime_r(&rawTime, &time);

    uint32_t length = sprintf(buffer, "%02d/%02d/%04d %02d:%02d:%02d.%03lu UTC: %s\r\n", time.tm_mday, MONTH_OFFSET + time.tm_mon, YEAR_OFFSET + time.tm_year, time.tm_hour, time.tm_min, time.tm_sec, currentMilliseconds, message);

    AudioMoth_writeToFile(buffer, length);

}

static GPS_fixResult_t setTimeFromGPS(bool showLED, AM_gpsFixMode_t fixMode, uint32_t timeout) {

    /* Enable GPS and get the current time */

    GPS_powerUpGPS();

    GPS_enableGPSInterface();

    uint32_t currentTime, currentMilliseconds;

    /* Open the GPS log file */

    bool success = AudioMoth_appendFile(GPS_FILENAME);

    /* Add power up message */

    static char *messages[] = {
        "GPS switched on for initial fix.",
        "GPS switched on for fix before recording period.",
        "GPS switched on for fix between recording periods.",
        "GPS switched on for fix after recording period.",
        "GPS switched on for fix before recording.",
        "GPS switched on for fix between recordings.",
        "GPS switched on for fix after recording."
    };

    char *message = messages[fixMode];

    AudioMoth_getTime(&currentTime, &currentMilliseconds);

    writeGPSLogMessage((char*)compressionBuffer, currentTime, currentMilliseconds, message);

    /* Set green LED and enter routine */

    gpsEnableLED = showLED;

    if (gpsEnableLED) AudioMoth_setGreenLED(true);

    GPS_fixResult_t result = GPS_setTimeFromGPS(timeout);

    AudioMoth_setRedLED(false);

    GPS_disableGPSInterface();

    /* Power down the GPS */

    uint32_t powerOffTime, powerOffMilliseconds;

    AudioMoth_getTime(&powerOffTime, &powerOffMilliseconds);

    if (result == GPS_SUCCESS || result == GPS_TIMEOUT) powerOffMilliseconds = 0;

    if (result == GPS_SUCCESS) {

        powerOffTime = MIN(timeSetFromGPS + GPS_ADDITIONAL_POWER_INTERVAL, timeout);

        while (currentTime < powerOffTime) {

            AudioMoth_feedWatchdog();

            AudioMoth_delay(SHORT_WAIT_INTERVAL);

            AudioMoth_getTime(&currentTime, &currentMilliseconds);

        }

    }

    AudioMoth_setGreenLED(false);

    GPS_powerDownGPS();

    /* Write result messages */

    if (result == GPS_CANCELLED_BY_MAGNETIC_SWITCH) writeGPSLogMessage((char*)compressionBuffer, powerOffTime, powerOffMilliseconds, "Time was not updated. Cancelled by magnetic switch.");

    if (result == GPS_CANCELLED_BY_SWITCH) writeGPSLogMessage((char*)compressionBuffer, powerOffTime, powerOffMilliseconds, "Time was not updated. Cancelled by switch position change.");

    if (result == GPS_TIMEOUT) writeGPSLogMessage((char*)compressionBuffer, powerOffTime, powerOffMilliseconds, "Time was not updated. Timed out.");

    writeGPSLogMessage((char*)compressionBuffer, powerOffTime, powerOffMilliseconds, "GPS switched off.\r\n");

    /* Close file and return */

    if (success) AudioMoth_closeFile();

    return result;

}

/* Magnetic switch wait functions */

static void startWaitingForMagneticSwitch() {

    /* Flash LED to indicate start of waiting for magnetic switch */

    FLASH_REPEAT_LED(Red, MAGNETIC_SWITCH_CHANGE_FLASHES, SHORT_LED_FLASH_DURATION);

    /* Cancel any scheduled recording */

    *indexOfNextRecording = 0;

    *timeOfNextRecording = UINT32_MAX;

    *startOfRecordingPeriod = UINT32_MAX;

    *durationOfNextRecording = UINT32_MAX;

    *timeOfNextGPSTimeSetting = UINT32_MAX;

    setBackupFlag(BACKUP_SHOULD_SET_TIME_FROM_GPS, false);

    setBackupFlag(BACKUP_WAITING_FOR_MAGNETIC_SWITCH, true);

}

static void stopWaitingForMagneticSwitch(uint32_t *currentTime, uint32_t *currentMilliseconds) {

    /* Flash LED to indicate end of waiting for magnetic switch */

    FLASH_REPEAT_LED(Green, MAGNETIC_SWITCH_CHANGE_FLASHES, SHORT_LED_FLASH_DURATION);

    /* Schedule next recording */

    AudioMoth_getTime(currentTime, currentMilliseconds);

    uint32_t scheduleTime = *currentTime + ROUNDED_UP_DIV(*currentMilliseconds + *recordingPreparationPeriod, MILLISECONDS_IN_SECOND);

    determineSunriseAndSunsetTimesAndScheduleRecording(scheduleTime);

    /* Update flag */

    setBackupFlag(BACKUP_WAITING_FOR_MAGNETIC_SWITCH, false);

}

/* Function to calculate the time to the next event */

static void calculateTimeToNextEvent(uint32_t currentTime, uint32_t currentMilliseconds, int64_t *timeUntilPreparationStart, int64_t *timeUntilNextGPSTimeSetting) {

    int64_t preparationPeriodBuffer = (configSettings->preparationPeriodBuffer == 0 ? DEFAULT_PREPARATION_PERIOD_BUFFER : configSettings->preparationPeriodBuffer * MILLISECONDS_IN_SECOND);

    *timeUntilPreparationStart = (int64_t)*timeOfNextRecording * MILLISECONDS_IN_SECOND - (int64_t)*recordingPreparationPeriod - preparationPeriodBuffer - (int64_t)currentTime * MILLISECONDS_IN_SECOND - (int64_t)currentMilliseconds;

    *timeUntilNextGPSTimeSetting = (int64_t)*timeOfNextGPSTimeSetting * MILLISECONDS_IN_SECOND - (int64_t)currentTime * MILLISECONDS_IN_SECOND - (int64_t)currentMilliseconds;

}

/* Main function */

int main(void) {

    /* Initialise device */

    AudioMoth_initialise();

#if STFT_DEBUG_MODE

    /* Bypass the normal recording schedule for the STFT timing test, but only when the switch is not in the USB position. If it is, fall through to the normal firmware flow below so AudioMoth_handleUSB() still runs and the device can always be reflashed via the switch, without needing the hardware bootloader trick. */

    if (AudioMoth_getSwitchPosition() != AM_SWITCH_USB) {

        runSTFTFeasibilityTest();

    }

#endif

#if TRIGGERED_RECORDING_MODE

    /* Same bypass pattern as STFT_DEBUG_MODE above - only when the switch is not in the USB position, so the device can always be reflashed via the switch. runTriggeredRecording() never returns while the switch stays in the recording position. */

    if (AudioMoth_getSwitchPosition() != AM_SWITCH_USB) {

        runTriggeredRecording();

    }

#endif

    AM_switchPosition_t switchPosition = AudioMoth_getSwitchPosition();

    if (AudioMoth_isInitialPowerUp()) {
        
        /* Initialise recording schedule variables */

        *indexOfNextRecording = 0;

        *timeOfNextRecording = UINT32_MAX;

        *startOfRecordingPeriod = UINT32_MAX;

        *durationOfNextRecording = UINT32_MAX;

        *timeOfNextGPSTimeSetting = UINT32_MAX;

        /* Initialise configuration writing variable */

        setBackupFlag(BACKUP_WRITTEN_CONFIGURATION_TO_FILE, false);

        /* Initialise recording state variables */

        *previousSwitchPosition = AM_SWITCH_NONE;

        setBackupFlag(BACKUP_MUST_SET_TIME_FROM_GPS, false);

        setBackupFlag(BACKUP_SHOULD_SET_TIME_FROM_GPS, false);

        setBackupFlag(BACKUP_READY_TO_MAKE_RECORDING, false);

        *numberOfRecordingErrors = 0;

        *numberOfConsecutiveRecordingErrors = 0;

        *recordingPreparationPeriod = INITIAL_PREPARATION_PERIOD;

        /* Initialise magnetic switch state variables */

        setBackupFlag(BACKUP_WAITING_FOR_MAGNETIC_SWITCH, false);

        /* Initialise the power down interval flag */

        setBackupFlag(BACKUP_POWERED_DOWN_WITH_SHORT_WAIT_INTERVAL, false);

        /* Initial GPS and sunrise and sunset variables */

        *gpsLastFixLatitude = 0;

        *gpsLastFixLongitude = 0;

        setBackupFlag(BACKUP_GPS_LOCATION_RECEIVED, false);

        setBackupFlag(BACKUP_ACOUSTIC_LOCATION_RECEIVED, false);

        setBackupFlag(BACKUP_ACOUSTIC_TIMEZONE_RECEIVED, false);

        *timeOfNextSunriseSunsetCalculation = 0;

        /* Copy default deployment ID */

        initialiseDeploymentID();

    }

    /* Check the persistent configuration firmware version and description matches the default values */

    if (memcmp(persistentConfigSettings->firmwareVersion, firmwareVersion, AM_FIRMWARE_VERSION_LENGTH) != 0 || memcmp(persistentConfigSettings->firmwareDescription, firmwareDescription, AM_FIRMWARE_DESCRIPTION_LENGTH) != 0) {

        memcpy(&tempPersistentConfigSettings->firmwareVersion, &firmwareVersion, AM_FIRMWARE_VERSION_LENGTH);

        memcpy(&tempPersistentConfigSettings->firmwareDescription, &firmwareDescription, AM_FIRMWARE_DESCRIPTION_LENGTH);

        memcpy(&tempPersistentConfigSettings->configSettings, (uint8_t*)&defaultConfigSettings, sizeof(configSettings_t));

        uint32_t numberOfBytes = ROUND_UP_TO_MULTIPLE(sizeof(persistentConfigSettings_t), UINT32_SIZE_IN_BYTES);

        bool success = AudioMoth_writeToFlashUserDataPage((uint8_t*)tempPersistentConfigSettings, numberOfBytes);

        if (success == false) {

            /* Cannot recover from a failure at this point */

            for (uint32_t i = 0; i < 5; i += 1) {

                AudioMoth_setBothLED(true);

                AudioMoth_delay(100);

                AudioMoth_setBothLED(false);

                AudioMoth_delay(100);

            }

            AudioMoth_powerDownAndWakeMilliseconds(DEFAULT_WAIT_INTERVAL);

        }

    }

    /* Handle the case that the switch is in USB position  */

    if (switchPosition == AM_SWITCH_USB) {

        if (configSettings->disableBatteryLevelDisplay == false && (*previousSwitchPosition == AM_SWITCH_DEFAULT || *previousSwitchPosition == AM_SWITCH_CUSTOM)) {

            flashLedToIndicateBatteryLife();

        }

        AudioMoth_handleUSB();

        SAVE_SWITCH_POSITION_AND_POWER_DOWN(DEFAULT_WAIT_INTERVAL);

    }

    /* Read the time */

    uint32_t currentTime;

    uint32_t currentMilliseconds;

    AudioMoth_getTime(&currentTime, &currentMilliseconds);

    /* Check if switch has just been moved to CUSTOM or DEFAULT */

    bool fileSystemEnabled = false;

    if (switchPosition != *previousSwitchPosition) {

        /* Reset the LED and GPS flags */

        *timeAtWhichToSwitchOffLED = UINT32_MAX;

        setBackupFlag(BACKUP_MUST_SET_TIME_FROM_GPS, false);

        setBackupFlag(BACKUP_SHOULD_SET_TIME_FROM_GPS, false);

        /* Reset the power down interval flag */

        setBackupFlag(BACKUP_POWERED_DOWN_WITH_SHORT_WAIT_INTERVAL, false);

        /* Check there are active recording periods if the switch is in CUSTOM position */

        setBackupFlag(BACKUP_READY_TO_MAKE_RECORDING, switchPosition == AM_SWITCH_DEFAULT || (switchPosition == AM_SWITCH_CUSTOM && (configSettings->activeRecordingPeriods > 0 || configSettings->enableSunRecording)));

        /* Check if acoustic configuration is required */

        if (getBackupFlag(BACKUP_READY_TO_MAKE_RECORDING)) {

            /* Determine if acoustic configuration is required */

            bool shouldPerformAcousticConfiguration = switchPosition == AM_SWITCH_CUSTOM && (AudioMoth_hasTimeBeenSet() == false || configSettings->requireAcousticConfiguration);

            /* Overrule this decision if setting of time from GPS is enabled and acoustic configuration not enforced. Also set GPS time setting flag */

            if (switchPosition == AM_SWITCH_CUSTOM && configSettings->enableTimeSettingFromGPS) {

                if (configSettings->requireAcousticConfiguration == false) shouldPerformAcousticConfiguration = false;

                setBackupFlag(BACKUP_MUST_SET_TIME_FROM_GPS, true);

            }

            /* Determine whether to listen for the acoustic tone */

            bool listenForAcousticTone = switchPosition == AM_SWITCH_CUSTOM && shouldPerformAcousticConfiguration == false;

            if (listenForAcousticTone) {

                AudioConfig_enableAudioConfiguration(configSettings->ignoreExternalMicrophoneForAcousticChime);

                shouldPerformAcousticConfiguration = AudioConfig_listenForAudioConfigurationTone(AUDIO_CONFIG_TONE_TIMEOUT);

            }

            if (shouldPerformAcousticConfiguration) {

                AudioMoth_setRedLED(true);

                AudioMoth_setGreenLED(false);

                audioConfigPulseCounter = 0;

                audioConfigStateLED = false;

                audioConfigToggleLED = false;

                acousticConfigurationPerformed = false;

                if (listenForAcousticTone == false) {

                    AudioConfig_enableAudioConfiguration(configSettings->ignoreExternalMicrophoneForAcousticChime);

                }

                AudioConfig_listenForAudioConfigurationPackets(listenForAcousticTone, AUDIO_CONFIG_PACKETS_TIMEOUT);

                AudioConfig_disableAudioConfiguration();

                if (acousticConfigurationPerformed) {

                    /* Indicate success with LED flashes */

                    AudioMoth_setRedLED(false);

                    AudioMoth_setGreenLED(true);

                    AudioMoth_delay(1000);
                    
                    AudioMoth_delay(1000);

                    AudioMoth_setGreenLED(false);

                    AudioMoth_delay(500);

                } else {

                    /* Turn off LED */

                    AudioMoth_setBothLED(false);

                    /* Determine if it is possible to still make a recording */

                    if (configSettings->requireAcousticConfiguration) {

                        setBackupFlag(BACKUP_READY_TO_MAKE_RECORDING, false);

                    } else {

                        setBackupFlag(BACKUP_READY_TO_MAKE_RECORDING, AudioMoth_hasTimeBeenSet() || getBackupFlag(BACKUP_MUST_SET_TIME_FROM_GPS));

                    }

                    /* Power down */

                    SAVE_SWITCH_POSITION_AND_POWER_DOWN(DEFAULT_WAIT_INTERVAL);

                }

            } else if (listenForAcousticTone) {

                AudioConfig_disableAudioConfiguration();

            }

        }

        /* Calculate time of next recording if ready to make a recording */

        if (getBackupFlag(BACKUP_READY_TO_MAKE_RECORDING)) {

            /* Enable energy saver mode */

            if (isEnergySaverMode(configSettings)) AudioMoth_setClockDivider(AM_HF_CLK_DIV2);

            /* Reset the recording error counter */

            *numberOfRecordingErrors = 0;

            *numberOfConsecutiveRecordingErrors = 0;

            /* Reset the recording preparation period to default */

            *recordingPreparationPeriod = INITIAL_PREPARATION_PERIOD;

            /* Reset persistent configuration write flag */

            setBackupFlag(BACKUP_WRITTEN_CONFIGURATION_TO_FILE, false);

            /* Reset GPS and sunrise and sunset variables */

            *gpsLastFixLatitude = 0;

            *gpsLastFixLongitude = 0;

            setBackupFlag(BACKUP_GPS_LOCATION_RECEIVED, false);

            *timeOfNextSunriseSunsetCalculation = 0;

            /* Try to write configuration now if it will not be written later when time is set */

            if (configSettings->enableSunRecording == false || getBackupFlag(BACKUP_MUST_SET_TIME_FROM_GPS) == false) {

                fileSystemEnabled = AudioMoth_enableFileSystem(configSettings->sampleRateDivider == 1 ? AM_SD_CARD_HIGH_SPEED : AM_SD_CARD_NORMAL_SPEED);

                if (fileSystemEnabled) {
                    
                    AudioMoth_getTime(&currentTime, &currentMilliseconds);

                    bool gpsLocationReceived = getBackupFlag(BACKUP_GPS_LOCATION_RECEIVED);

                    bool acousticLocationReceived = getBackupFlag(BACKUP_ACOUSTIC_LOCATION_RECEIVED);

                    bool success = writeConfigurationToFile((char*)compressionBuffer, configSettings, currentTime, gpsLocationReceived, gpsLatitude, gpsLongitude, acousticLocationReceived, acousticLatitude, acousticLongitude, firmwareDescription, firmwareVersion, (uint8_t*)AM_UNIQUE_ID_START_ADDRESS, deploymentID, defaultDeploymentID);

                    setBackupFlag(BACKUP_WRITTEN_CONFIGURATION_TO_FILE, success);

                }

            }

            /* Update the time and calculate earliest schedule start time */

            AudioMoth_getTime(&currentTime, &currentMilliseconds);

            uint32_t scheduleTime = currentTime + ROUNDED_UP_DIV(currentMilliseconds + *recordingPreparationPeriod, MILLISECONDS_IN_SECOND);

            /* Schedule the next recording */

            if (switchPosition == AM_SWITCH_CUSTOM) {

                setBackupFlag(BACKUP_WAITING_FOR_MAGNETIC_SWITCH, configSettings->enableMagneticSwitch);

                if (configSettings->enableMagneticSwitch || getBackupFlag(BACKUP_MUST_SET_TIME_FROM_GPS)) {

                    *indexOfNextRecording = 0;

                    *timeOfNextRecording = UINT32_MAX;

                    *startOfRecordingPeriod = UINT32_MAX;

                    *durationOfNextRecording = UINT32_MAX;

                    *timeOfNextGPSTimeSetting = UINT32_MAX;

                } else {

                    *timeOfNextRecording = UINT32_MAX;

                    *startOfRecordingPeriod = UINT32_MAX;

                    determineSunriseAndSunsetTimesAndScheduleRecording(scheduleTime);

                    if (configSettings->enableLED == false) *timeAtWhichToSwitchOffLED = currentTime + SECONDS_IN_MINUTE;

                }

            }

            /* Set parameters to start recording now */

            if (switchPosition == AM_SWITCH_DEFAULT) {

                *indexOfNextRecording = 0; 

                *timeOfNextRecording = scheduleTime;

                *startOfRecordingPeriod = scheduleTime;

                *durationOfNextRecording = UINT32_MAX;

                *timeOfNextGPSTimeSetting = UINT32_MAX;

            }

        }

    }
    
    /* If not ready to make a recording then flash LED and power down */

    if (getBackupFlag(BACKUP_READY_TO_MAKE_RECORDING) == false) {

        FLASH_LED(Both, SHORT_LED_FLASH_DURATION)

        SAVE_SWITCH_POSITION_AND_POWER_DOWN(DEFAULT_WAIT_INTERVAL);

    }

    /* Enable the magnetic switch */

    bool magneticSwitchEnabled = configSettings->enableMagneticSwitch && switchPosition == AM_SWITCH_CUSTOM;

    if (magneticSwitchEnabled) GPS_enableMagneticSwitch();

    /* Reset LED flags */

    enableLED = currentTime < *timeAtWhichToSwitchOffLED;

    bool shouldSuppressLED = getBackupFlag(BACKUP_POWERED_DOWN_WITH_SHORT_WAIT_INTERVAL);

    setBackupFlag(BACKUP_POWERED_DOWN_WITH_SHORT_WAIT_INTERVAL, false);

    /* Calculate time until next activity */

    int64_t timeUntilPreparationStart;
    
    int64_t timeUntilNextGPSTimeSetting;

    calculateTimeToNextEvent(currentTime, currentMilliseconds, &timeUntilPreparationStart, &timeUntilNextGPSTimeSetting);

    /* If the GPS synchronisation window has passed then cancel it */

    int64_t timeSinceScheduledGPSTimeSetting = -timeUntilNextGPSTimeSetting;

    uint32_t gpsTimeSettingPeriod = configSettings->gpsTimeSettingPeriod == 0 ? GPS_DEFAULT_TIME_SETTING_PERIOD : configSettings->gpsTimeSettingPeriod * SECONDS_IN_MINUTE;

    if (timeSinceScheduledGPSTimeSetting > gpsTimeSettingPeriod * MILLISECONDS_IN_SECOND) {

        *timeOfNextGPSTimeSetting = UINT32_MAX;

        calculateTimeToNextEvent(currentTime, currentMilliseconds, &timeUntilPreparationStart, &timeUntilNextGPSTimeSetting);

    }

    /* Set the time from the GPS */

    if (getBackupFlag(BACKUP_MUST_SET_TIME_FROM_GPS) && getBackupFlag(BACKUP_WAITING_FOR_MAGNETIC_SWITCH) == false) {

        /* Enable the file system and set the time from the GPS */

        if (!fileSystemEnabled) fileSystemEnabled = AudioMoth_enableFileSystem(AM_SD_CARD_NORMAL_SPEED);
            
        GPS_fixResult_t fixResult = setTimeFromGPS(enableLED, INITIAL_GPS_FIX, currentTime + GPS_INITIAL_TIME_SETTING_PERIOD);

        /* Update the schedule if successful */

        if (fixResult == GPS_SUCCESS) {

            /* Reset the flag */

            setBackupFlag(BACKUP_MUST_SET_TIME_FROM_GPS, false);

            /* Update the GPS location */

            setBackupFlag(BACKUP_GPS_LOCATION_RECEIVED, true);

            *gpsLatitude = *gpsLastFixLatitude;

            *gpsLongitude = *gpsLastFixLongitude;

            /* Write the configuration file */

            if (configSettings->enableSunRecording && fileSystemEnabled) {

                AudioMoth_getTime(&currentTime, &currentMilliseconds);

                bool gpsLocationReceived = getBackupFlag(BACKUP_GPS_LOCATION_RECEIVED);

                bool acousticLocationReceived = getBackupFlag(BACKUP_ACOUSTIC_LOCATION_RECEIVED);

                bool success = writeConfigurationToFile((char*)compressionBuffer, configSettings, currentTime, gpsLocationReceived, gpsLatitude, gpsLongitude, acousticLocationReceived, acousticLatitude, acousticLongitude, firmwareDescription, firmwareVersion, (uint8_t*)AM_UNIQUE_ID_START_ADDRESS, deploymentID, defaultDeploymentID);

                setBackupFlag(BACKUP_WRITTEN_CONFIGURATION_TO_FILE, success);

            }

            /* Schedule the next recording */

            AudioMoth_getTime(&currentTime, &currentMilliseconds);

            uint32_t scheduleTime = currentTime + ROUNDED_UP_DIV(currentMilliseconds + *recordingPreparationPeriod, MILLISECONDS_IN_SECOND);

            determineSunriseAndSunsetTimesAndScheduleRecording(scheduleTime);

            if (configSettings->enableLED == false) *timeAtWhichToSwitchOffLED = currentTime + SECONDS_IN_MINUTE;

        }
        
        /* If time setting was cancelled with the magnet switch then start waiting for the magnetic switch */

        if (fixResult == GPS_CANCELLED_BY_MAGNETIC_SWITCH) startWaitingForMagneticSwitch();

        /* Power down */

        SAVE_SWITCH_POSITION_AND_POWER_DOWN(DEFAULT_WAIT_INTERVAL);

    }

    /* Make a recording */
    
    if (timeUntilPreparationStart <= 0) {

        /* Enable energy saver mode */

        if (isEnergySaverMode(configSettings)) AudioMoth_setClockDivider(AM_HF_CLK_DIV2);

        /* Write configuration if not already done so */

        if (getBackupFlag(BACKUP_WRITTEN_CONFIGURATION_TO_FILE) == false) {

            if (!fileSystemEnabled) fileSystemEnabled = AudioMoth_enableFileSystem(configSettings->sampleRateDivider == 1 ? AM_SD_CARD_HIGH_SPEED : AM_SD_CARD_NORMAL_SPEED);

            if (fileSystemEnabled) {

                bool gpsLocationReceived = getBackupFlag(BACKUP_GPS_LOCATION_RECEIVED);

                bool acousticLocationReceived = getBackupFlag(BACKUP_ACOUSTIC_LOCATION_RECEIVED);

                bool success = writeConfigurationToFile((char*)compressionBuffer, configSettings, currentTime, gpsLocationReceived, gpsLatitude, gpsLongitude, acousticLocationReceived, acousticLatitude, acousticLongitude, firmwareDescription, firmwareVersion, (uint8_t*)AM_UNIQUE_ID_START_ADDRESS, deploymentID, defaultDeploymentID);

                setBackupFlag(BACKUP_WRITTEN_CONFIGURATION_TO_FILE, success);

            }

        }

        /* Make the recording */

        uint32_t fileOpenTime;

        uint32_t fileOpenMilliseconds;

        AM_recordingState_t recordingState = RECORDING_OKAY;

        /* Measure battery voltage */

        uint32_t supplyVoltage = AudioMoth_getSupplyVoltage();

        AM_extendedBatteryState_t extendedBatteryState = AudioMoth_getExtendedBatteryState(supplyVoltage);

        /* Check if low voltage check is enabled and that the voltage is okay */

        bool okayToMakeRecording = true;

        if (configSettings->enableLowVoltageCutoff) {

            AudioMoth_enableSupplyMonitor();

            AudioMoth_setSupplyMonitorThreshold(MINIMUM_SUPPLY_VOLTAGE);

            okayToMakeRecording = AudioMoth_isSupplyAboveThreshold();

        }

        /* Make recording if okay */

        if (okayToMakeRecording) {

            AudioMoth_enableTemperature();

            int32_t temperature = AudioMoth_getTemperature();

            AudioMoth_disableTemperature();

            if (!fileSystemEnabled) fileSystemEnabled = AudioMoth_enableFileSystem(configSettings->sampleRateDivider == 1 ? AM_SD_CARD_HIGH_SPEED : AM_SD_CARD_NORMAL_SPEED);

            if (fileSystemEnabled)  {

                recordingState = makeRecording(*timeOfNextRecording, *durationOfNextRecording, *timeAtWhichToSwitchOffLED, extendedBatteryState, temperature, &fileOpenTime, &fileOpenMilliseconds);

            } else {

                if (enableLED) FLASH_LED(Both, LONG_LED_FLASH_DURATION);

                recordingState = SDCARD_WRITE_ERROR;

            }

        } else {

            recordingState = SUPPLY_VOLTAGE_LOW;

        }

        /* Disable low voltage monitor if it was used */

        if (configSettings->enableLowVoltageCutoff) AudioMoth_disableSupplyMonitor();

        /* Enable the error warning flashes */

        if (recordingState == SUPPLY_VOLTAGE_LOW) {

            AudioMoth_delay(LONG_LED_FLASH_DURATION);

            if (enableLED) FLASH_LED(Both, LONG_LED_FLASH_DURATION);

            *numberOfRecordingErrors += 1;

            *numberOfConsecutiveRecordingErrors += 1;

        } else if (recordingState == SDCARD_WRITE_ERROR) {

            *numberOfRecordingErrors += 1;

            *numberOfConsecutiveRecordingErrors += 1;

        } else {

            *numberOfConsecutiveRecordingErrors = 0;

        }

        /* Update the preparation period */

        if (recordingState == RECORDING_OKAY) {

            int64_t measuredPreparationPeriod = (int64_t)fileOpenTime * MILLISECONDS_IN_SECOND + (int64_t)fileOpenMilliseconds - (int64_t)currentTime * MILLISECONDS_IN_SECOND - (int64_t)currentMilliseconds;

            measuredPreparationPeriod = MIN(MAXIMUM_PREPARATION_PERIOD, MAX(MINIMUM_PREPARATION_PERIOD, measuredPreparationPeriod));

            if (*recordingPreparationPeriod == INITIAL_PREPARATION_PERIOD) {

                *recordingPreparationPeriod = measuredPreparationPeriod;

            } else {

                *recordingPreparationPeriod = ROUNDED_DIV(*recordingPreparationPeriod * (PREPARATION_PERIOD_SMOOTHING_FACTOR - 1) + measuredPreparationPeriod, PREPARATION_PERIOD_SMOOTHING_FACTOR);

            }

        }

        /* Update the time and calculate earliest schedule start time */

        AudioMoth_getTime(&currentTime, &currentMilliseconds);

        uint32_t scheduleTime = currentTime + ROUNDED_UP_DIV(currentMilliseconds + *recordingPreparationPeriod, MILLISECONDS_IN_SECOND);

        /* Schedule the next recording */

        if (*numberOfConsecutiveRecordingErrors >= MAX_CONSECUTIVE_RECORDING_ERRORS) {

            /* Cancel the schedule */

            *indexOfNextRecording = 0; 

            *timeOfNextRecording = UINT32_MAX;

            *startOfRecordingPeriod = UINT32_MAX;

            *durationOfNextRecording = UINT32_MAX;

            *timeOfNextGPSTimeSetting = UINT32_MAX;

        } else if (switchPosition == AM_SWITCH_CUSTOM) {

            /* Update schedule time as if the recording has ended correctly */

            if (recordingState == RECORDING_OKAY || recordingState == SUPPLY_VOLTAGE_LOW || recordingState == SDCARD_WRITE_ERROR) {

                scheduleTime = MAX(scheduleTime, *timeOfNextRecording + *durationOfNextRecording);

            }

            /* Calculate the next recording schedule */

            determineSunriseAndSunsetTimesAndScheduleRecording(scheduleTime);

        } else {

            /* Set parameters to start recording now */

            *indexOfNextRecording = 0; 

            *timeOfNextRecording = scheduleTime;

            *startOfRecordingPeriod = scheduleTime;

            *durationOfNextRecording = UINT32_MAX;

            *timeOfNextGPSTimeSetting = UINT32_MAX;

        }

        /* If recording was cancelled with the magnetic switch then start waiting for the magnetic switch */

        if (recordingState == MAGNETIC_SWITCH) {
            
            startWaitingForMagneticSwitch();

            SAVE_SWITCH_POSITION_AND_POWER_DOWN(DEFAULT_WAIT_INTERVAL);

        }

         /* Power down with short interval if the next recording is due */

        if (switchPosition == AM_SWITCH_CUSTOM) {

            calculateTimeToNextEvent(currentTime, currentMilliseconds, &timeUntilPreparationStart, &timeUntilNextGPSTimeSetting);

            if (timeUntilPreparationStart < DEFAULT_WAIT_INTERVAL) {

                setBackupFlag(BACKUP_POWERED_DOWN_WITH_SHORT_WAIT_INTERVAL, true);

                SAVE_SWITCH_POSITION_AND_POWER_DOWN(SHORT_WAIT_INTERVAL);

            }
        
        }

        /* Power down */

        SAVE_SWITCH_POSITION_AND_POWER_DOWN(DEFAULT_WAIT_INTERVAL);

    }

    /* Update the time from the GPS */
    
    if ((timeUntilNextGPSTimeSetting <= 0 || getBackupFlag(BACKUP_SHOULD_SET_TIME_FROM_GPS)) && timeUntilPreparationStart > GPS_MIN_TIME_SETTING_PERIOD * MILLISECONDS_IN_SECOND) {

        /* Set the time from the GPS */

        AudioMoth_enableFileSystem(AM_SD_CARD_NORMAL_SPEED);

        AM_gpsFixMode_t fixMode = configSettings->enableTimeSettingBeforeAndAfterRecordings ? GPS_FIX_BEFORE_INDIVIDUAL_RECORDING : GPS_FIX_BEFORE_RECORDING_PERIOD;
        
        if (getBackupFlag(BACKUP_SHOULD_SET_TIME_FROM_GPS)) fixMode += 1;

        if (timeUntilNextGPSTimeSetting > 0) fixMode += 1;

        uint32_t gpsTimeSettingPeriod = configSettings->gpsTimeSettingPeriod == 0 ? GPS_DEFAULT_TIME_SETTING_PERIOD : configSettings->gpsTimeSettingPeriod * SECONDS_IN_MINUTE;

        uint32_t timeOut = MIN(*timeOfNextRecording - ROUNDED_UP_DIV(*recordingPreparationPeriod, MILLISECONDS_IN_SECOND) - GPS_TIME_SETTING_MARGIN, currentTime + gpsTimeSettingPeriod);

        GPS_fixResult_t fixResult = setTimeFromGPS(enableLED, fixMode, timeOut);

        /* Update flag and next scheduled GPS fix time */

        if (timeUntilNextGPSTimeSetting <= 0) *timeOfNextGPSTimeSetting = UINT32_MAX;

        setBackupFlag(BACKUP_SHOULD_SET_TIME_FROM_GPS, false);

        /* If time setting was cancelled with the magnet switch then start waiting for the magnetic switch */

        if (fixResult == GPS_CANCELLED_BY_MAGNETIC_SWITCH) startWaitingForMagneticSwitch();

        /* Power down */

        SAVE_SWITCH_POSITION_AND_POWER_DOWN(DEFAULT_WAIT_INTERVAL);

    } 

    /* Update the post-recording GPS flag if the previous condition was not met */

    setBackupFlag(BACKUP_SHOULD_SET_TIME_FROM_GPS, false);

    /* Power down if switch position has changed */

    if (switchPosition != *previousSwitchPosition) SAVE_SWITCH_POSITION_AND_POWER_DOWN(DEFAULT_WAIT_INTERVAL);

    /* Calculate the wait intervals */

    int64_t waitIntervalMilliseconds = WAITING_LED_FLASH_INTERVAL;

    uint32_t waitIntervalSeconds = WAITING_LED_FLASH_INTERVAL / MILLISECONDS_IN_SECOND;

    if (getBackupFlag(BACKUP_WAITING_FOR_MAGNETIC_SWITCH)) {
        
        waitIntervalMilliseconds *= MAGNETIC_SWITCH_WAIT_MULTIPLIER;

        waitIntervalSeconds *= MAGNETIC_SWITCH_WAIT_MULTIPLIER;

    }

    /* Wait for the next event whilst flashing the LED */

    bool startedRealTimeClock = false;

    while (true) {

        /* Update the time */

        AudioMoth_getTime(&currentTime, &currentMilliseconds);

        /* Handle magnetic switch event */

        bool magneticSwitchEvent = magneticSwitch || (magneticSwitchEnabled && GPS_isMagneticSwitchClosed());

        if (magneticSwitchEvent) {

            if (getBackupFlag(BACKUP_WAITING_FOR_MAGNETIC_SWITCH)) {

                stopWaitingForMagneticSwitch(&currentTime, &currentMilliseconds);

                if (switchPosition == AM_SWITCH_CUSTOM) {

                    if (configSettings->enableTimeSettingFromGPS) {
                        
                        setBackupFlag(BACKUP_MUST_SET_TIME_FROM_GPS, true);

                    } else if (configSettings->enableLED == false) {
                        
                        *timeAtWhichToSwitchOffLED = currentTime + SECONDS_IN_MINUTE;

                    }

                }

            } else {

                startWaitingForMagneticSwitch();

            }
            
            SAVE_SWITCH_POSITION_AND_POWER_DOWN(DEFAULT_WAIT_INTERVAL);

        }

        /* Handle switch position change */

        bool switchEvent = switchPositionChanged || AudioMoth_getSwitchPosition() != *previousSwitchPosition;
       
        if (switchEvent) SAVE_SWITCH_POSITION_AND_POWER_DOWN(DEFAULT_WAIT_INTERVAL);
        
        /* Calculate the time to the next event */
        
        calculateTimeToNextEvent(currentTime, currentMilliseconds, &timeUntilPreparationStart, &timeUntilNextGPSTimeSetting);

        int64_t timeToEarliestEvent = MIN(timeUntilPreparationStart, timeUntilNextGPSTimeSetting);

        /* Flash LED */

        enableLED = currentTime < *timeAtWhichToSwitchOffLED;

        bool shouldFlashLED = getBackupFlag(BACKUP_WAITING_FOR_MAGNETIC_SWITCH) || (enableLED && shouldSuppressLED == false && timeToEarliestEvent > MINIMUM_LED_FLASH_INTERVAL);

        if (shouldFlashLED) {

            if (*numberOfRecordingErrors > 0) {

                FLASH_LED(Both, WAITING_LED_FLASH_DURATION);

            } else {

                FLASH_LED(Green, WAITING_LED_FLASH_DURATION);

            }

        }

        /* Check there is time to sleep */

        if (timeToEarliestEvent < waitIntervalMilliseconds) {
            
            /* Calculate the remaining time to power down */

            uint32_t timeToWait = timeToEarliestEvent < 0 ? 0 : timeToEarliestEvent;

            SAVE_SWITCH_POSITION_AND_POWER_DOWN(timeToWait);

        }

        /* Start the real time clock if it isn't running */

        if (startedRealTimeClock == false) {

            AudioMoth_startRealTimeClock(waitIntervalSeconds);

            startedRealTimeClock = true;

        }

        /* Enter deep sleep */

        AudioMoth_deepSleep();

        /* Handle time overflow on awakening */

        AudioMoth_checkAndHandleTimeOverflow();

    }

}

/* Time zone handler */

inline void AudioMoth_timezoneRequested(int32_t *timezoneHours, int32_t *timezoneMinutes) {

    if (configSettings->useTimezoneFromAcousticChime && getBackupFlag(BACKUP_ACOUSTIC_TIMEZONE_RECEIVED)) {

        *timezoneHours = *acousticTimezoneMinutes / MINUTES_IN_HOUR;
    
        *timezoneMinutes = *acousticTimezoneMinutes % MINUTES_IN_HOUR;

    } else {

        *timezoneHours = configSettings->timezoneHours;

        *timezoneMinutes = configSettings->timezoneMinutes;

    }

}

/* GPS time handlers */

inline void GPS_handleSetTime(uint32_t time, uint32_t milliseconds, int64_t timeDifference, uint32_t measuredClockFrequency) {

    /* Update the time if appropriate */

    timeSetFromGPS = time;

    if (!AudioMoth_hasTimeBeenSet()) {

        AudioMoth_setTime(time, milliseconds);

        writeGPSLogMessage((char*)compressionBuffer, time, milliseconds, "Time was set from GPS.");

    } else {

        int64_t maximumTimeDifference = (int64_t)GPS_MAXIMUM_DIFFERENCE * (int64_t)MICROSECONDS_IN_SECOND;

        if (timeDifference == 0) {

            writeGPSLogMessage((char*)compressionBuffer, time, milliseconds, "Time was not updated. The internal clock was correct.");

        } else if (timeDifference < -maximumTimeDifference || timeDifference > maximumTimeDifference) {

            writeGPSLogMessage((char*)compressionBuffer, time, milliseconds, "Time was not updated. The discrepancy between the internal clock and the GPS was too large.");

        } else {

            AudioMoth_setTime(time, milliseconds);

            int32_t scaledTimeDifference = (int32_t)ROUNDED_DIV(timeDifference, MICROSECONDS_IN_MILLISECOND / 10);

            bool isFast = scaledTimeDifference > 0;

            uint32_t units = ABS(scaledTimeDifference) / 10;

            uint32_t tenths = ABS(scaledTimeDifference) % 10;
            
            sprintf(gpsMessageBuffer, "Time was updated. The internal clock was %ld.%ldms %s.", units, tenths, isFast ? "fast" : "slow");

            writeGPSLogMessage((char*)compressionBuffer, time, milliseconds, gpsMessageBuffer);

        }

    }

    /* Calculate the actual sampling rate */

    uint32_t intendedClockFrequency = AudioMoth_getClockFrequency();

    uint32_t intendedSamplingRate = configSettings->sampleRate / configSettings->sampleRateDivider;

    uint32_t clockTicksPerSample = intendedClockFrequency / intendedSamplingRate;

    uint64_t actualSamplingRate = ROUNDED_DIV(GPS_FREQUENCY_PRECISION * (uint64_t)measuredClockFrequency, (uint64_t)clockTicksPerSample);

    uint32_t integerPart = actualSamplingRate / GPS_FREQUENCY_PRECISION;

    uint32_t fractionalPart = actualSamplingRate % GPS_FREQUENCY_PRECISION;

    sprintf(gpsMessageBuffer, "Actual sample rate will be %lu.%03lu Hz.", integerPart, fractionalPart);

    writeGPSLogMessage((char*)compressionBuffer, time, milliseconds, gpsMessageBuffer);

}

inline void GPS_handleGetTime(uint32_t *time, uint32_t *milliseconds) {

    AudioMoth_getTime(time, milliseconds);

}

inline void GPS_handleGetTimeMicroseconds(uint32_t *time, uint32_t *microseconds) {

    AudioMoth_getTimeMicroseconds(time, microseconds);

}

/* GPS format conversion */

static int32_t convertToDecimalDegrees(uint32_t degrees, uint32_t minutes, uint32_t tenThousandths, char direction) {

    int32_t value = degrees * GPS_LOCATION_PRECISION;

    value += ROUNDED_DIV(minutes * GPS_LOCATION_PRECISION, MINUTES_IN_DEGREE);

    value += ROUNDED_DIV(tenThousandths * (GPS_LOCATION_PRECISION / 10000), MINUTES_IN_DEGREE); 

    if (direction == 'S' || direction == 'W') value *= -1;

    return value;

}

/* GPS interrupt handlers */

inline void GPS_handleTickEvent() {

    if (gpsTickEventCount == 0) {

        gpsTickEventModulo = GPS_TICK_EVENTS_PER_SECOND;
    
        if (gpsPPSEvent || gpsFixEvent) {

            gpsTickEventModulo /= 3;

        } else if (gpsMessageEvent) {

            gpsTickEventModulo /= 2;

        }

        gpsMessageEvent = false;

        gpsFixEvent = false;

        gpsPPSEvent = false;

    }

    if (gpsEnableLED) AudioMoth_setRedLED(gpsTickEventCount % gpsTickEventModulo == 0);

    gpsTickEventCount = (gpsTickEventCount + 1) % GPS_TICK_EVENTS_PER_SECOND;

}

inline void GPS_handlePPSEvent(uint32_t time, uint32_t milliseconds) {

    writeGPSLogMessage((char*)compressionBuffer, time, milliseconds, "Received pulse per second signal.");

    gpsPPSEvent = true;

}

inline void GPS_handleFixEvent(uint32_t time, uint32_t milliseconds, GPS_fixTime_t *fixTime, GPS_fixPosition_t *fixPosition, char *message) {

    *gpsLastFixLatitude = convertToDecimalDegrees(fixPosition->latitudeDegrees, fixPosition->latitudeMinutes, fixPosition->latitudeTenThousandths, fixPosition->latitudeDirection);

    *gpsLastFixLongitude = convertToDecimalDegrees(fixPosition->longitudeDegrees, fixPosition->longitudeMinutes, fixPosition->longitudeTenThousandths, fixPosition->longitudeDirection);

    uint32_t length = sprintf(gpsMessageBuffer, "Received GPS fix - %ld.%06ld°%c %ld.%06ld°%c ", ABS(*gpsLastFixLatitude) / GPS_LOCATION_PRECISION, ABS(*gpsLastFixLatitude) % GPS_LOCATION_PRECISION, fixPosition->latitudeDirection, ABS(*gpsLastFixLongitude) / GPS_LOCATION_PRECISION, ABS(*gpsLastFixLongitude) % GPS_LOCATION_PRECISION, fixPosition->longitudeDirection);
    
    sprintf(gpsMessageBuffer + length, "at %02u/%02u/%04u %02u:%02u:%02u.%03u UTC.", fixTime->day, fixTime->month, fixTime->year, fixTime->hours, fixTime->minutes, fixTime->seconds, fixTime->milliseconds);

    writeGPSLogMessage((char*)compressionBuffer, time, milliseconds, gpsMessageBuffer);

    gpsFixEvent = true;

}

inline void GPS_handleMessageEvent(uint32_t time, uint32_t milliseconds, char *message) {

    if (!gpsFirstMessageReceived) {

        writeGPSLogMessage((char*)compressionBuffer, time, milliseconds, "Received first GPS message.");

        gpsFirstMessageReceived = true;

    }

    gpsMessageEvent = true;

}

inline void GPS_handleMagneticSwitchInterrupt() {

    magneticSwitch = true;

    GPS_cancelTimeSetting(GPS_CANCEL_BY_MAGNETIC_SWITCH);

}

/* AudioMoth interrupt handlers */

inline void AudioMoth_handleMicrophoneChangeInterrupt() {

    microphoneChanged = true;

    AudioConfig_handleMicrophoneChange();

}

inline void AudioMoth_handleSwitchInterrupt() {

    switchPositionChanged = true;

    AudioConfig_cancelAudioConfiguration();

    GPS_cancelTimeSetting(GPS_CANCEL_BY_SWITCH);

}

#if STFT_DEBUG_MODE || TRIGGERED_RECORDING_MODE

/* STFT feasibility test parameters - buffer is 1024 samples at 16kHz (64ms). Shared by
   STFT_DEBUG_MODE (runSTFTFeasibilityTest()) and TRIGGERED_RECORDING_MODE
   (runTriggeredRecording()) - the STFT/RMS/model-inference pipeline itself
   (computeSpectrogram()/computeBufferGain()/resetRMSNormalization()) is identical for both;
   only the parts specific to STFT_DEBUG_MODE's own CSV/BINARY debug-file output (further
   below) are still gated on STFT_DEBUG_MODE alone. */

#define STFT_SAMPLE_RATE                         16000

#define STFT_FFT_LENGTH                          256

#define STFT_HOP_LENGTH                          128

#define STFT_NUMBER_OF_FRAMES                    ((MAXIMUM_SAMPLES_IN_DMA_TRANSFER - STFT_FFT_LENGTH) / STFT_HOP_LENGTH + 1)

#define STFT_NUMBER_OF_BINS                       (STFT_FFT_LENGTH / 2 + 1)

#define STFT_LOG_EPSILON                          1.0e-6f

/* Matches student_model.py's LogSpectrogram exactly: bins below 150Hz are zeroed (not just
   attenuated) before the log, a brick-wall cut to discard wind/handling rumble and DC offset
   that would otherwise show up as high log-magnitude energy the model was never trained to
   see there. bin_width = STFT_SAMPLE_RATE / STFT_FFT_LENGTH = 62.5Hz; highpass_bin =
   ceil(150 / 62.5) = 3, so bins 0/1/2 (0Hz/62.5Hz/125Hz) are masked, matching
   math.ceil(150.0 / bin_width) there bit for bit - only valid for this exact sample
   rate/FFT length pair, not recomputed generically. */

#define STFT_HIGHPASS_BIN                         3

/* Streaming RMS normalization, matching dataset.streaming_rms_normalize() exactly (see
   test-audiomoth-cnn/dataset.py/CLAUDE notes) - training's old rms_normalize() needed the
   RMS of the entire 3-second clip before it could normalize a single sample, something a
   real-time device can never do (it would have to know the future). This is the causal
   equivalent: an exponential moving average of each buffer's own mean-square power, updated
   once per buffer, turned into a linear gain applied to that buffer's samples before
   windowing/FFT (folded directly into the existing per-sample window multiply below, never
   mutating the raw DMA buffer itself). Persists for the whole recording - reset only once,
   at the same point ModelInference_reset() resets the LSTM state (see
   resetRMSNormalization()/runSTFTFeasibilityTest()), never mid-recording.

   STFT_INT16_FULL_SCALE: the firmware's own raw ADC samples are never divided by 32768 (see
   the STFT_LOG_MODE branches below - source[...] is cast straight to float), so this works
   in that same raw scale, not a [-1,1]-normalized one.
   STFT_RISE_EMA_BUFFERS: 2/(N+1) is the standard convention for "this exponential average
   behaves like a simple moving average of N samples" - kept at the original ~3s time constant
   (MAXIMUM_SAMPLES_IN_DMA_TRANSFER=1024 samples per buffer @ 16kHz = 64ms/buffer,
   46.875 buffers = 3.0s) - not empirically tuned, must stay equal to dataset.py's
   RISE_EMA_BUFFERS if that ever changes.

   Dual time constant: slow to rise, fast to fall - the OPPOSITE asymmetry from a classic
   peak-limiting AGC (fast attack/slow release). This normalizer levels speech for detection,
   not peak-limits for clipping protection: a fast rise would react to the detected speech
   itself (which elevates bufferMeanSq), dragging the gain down while the speech is playing
   and self-suppressing the very signal the model needs to see; a slow fall leaves the gain
   suppressed for ~3s after a loud event ends, hurting detection of whatever quiet/normal
   content follows immediately after - a failure mode observed on real recordings.

   STFT_FALL_EMA_BUFFERS was originally 8 (~0.5s), validated only against a synthetic loud-
   burst-then-quiet step function. On real audio (continuously fluctuating, not a step
   function), "quieter than the running estimate" is true on ~57% of buffers, not just "just
   after a loud event ended" - at 8 buffers that fired large gain swings on more than a
   quarter of all buffers (>10% buffer-to-buffer change), an AGC "pumping" artifact riding
   the signal's natural envelope far more aggressively than intended, plausibly washing out
   amplitude-contrast cues the model relies on (observed as barely-moved frame-level
   precision@recall>=95% after retraining with this normalizer, and reduced real-world
   sensitivity even after re-tuning TRIGGER_PROBABILITY_THRESHOLD - see EXPERIMENTS.md). 16
   buffers (~1s) cuts buffers with a >10% gain jump from ~27% to ~5% on real audio while
   still recovering to ~90% of target within ~1.9s of a loud event ending - not a proven-
   optimal value, but a meaningfully less pumpy one. Must stay equal to dataset.py's
   RISE_EMA_BUFFERS/FALL_EMA_BUFFERS if either ever changes. */

/* Mirrors test-audiomoth-cnn/config.yaml's audio.normalization - "ema" (below), "none" (no
   gain applied at all, raw ADC samples straight into the STFT - a baseline to check whether
   normalizing loudness helps at all, now that the dataset itself already varies volume via
   real per-segment SNR), or "pcen" (Per-Channel Energy Normalization, Wang et al. 2017 - an
   adaptive gain computed independently per frequency bin, on the spectrogram itself, AFTER
   the FFT, instead of one global gain applied to the waveform BEFORE it - see
   test-audiomoth-cnn/student_model.py's LearnablePCEN and this file's own stftFastPow/
   computeSpectrogram below). Compile-time, not runtime-configurable - same pattern as
   STFT_DEBUG_MODE/TRIGGERED_RECORDING_MODE above.

   PCEN's own per-band alpha/delta/r/ema_alpha are NOT compile-time constants shared across
   every checkpoint the way STFT_RISE_EMA_ALPHA etc. are - they're LEARNED per band by a given
   training run (see LearnablePCEN) and baked into model/pcen.h/pcen.c by
   test-audiomoth-cnn/export_pcen.py, the same "regenerate, don't hand-edit, copy alongside
   model.h/model.c" convention model.h/model.c already use. A pcen.h/pcen.c mismatched to the
   model.h/model.c it's paired with (e.g. from two different checkpoints) is a silent
   train/inference skew, exactly like mismatched STFT constants would be. */

#define STFT_NORMALIZATION_EMA                    0
#define STFT_NORMALIZATION_NONE                   1
#define STFT_NORMALIZATION_PCEN                   2

#define STFT_NORMALIZATION_MODE                   STFT_NORMALIZATION_PCEN

#define STFT_TARGET_RMS                           0.1f
#define STFT_INT16_FULL_SCALE                     32768.0f
#define STFT_RISE_EMA_BUFFERS                     46.875f
#define STFT_FALL_EMA_BUFFERS                     16.0f
#define STFT_RISE_EMA_ALPHA                       (2.0f / (STFT_RISE_EMA_BUFFERS + 1.0f))
#define STFT_FALL_EMA_ALPHA                       (2.0f / (STFT_FALL_EMA_BUFFERS + 1.0f))

#define STFT_TWO_PI                               6.28318530717958647692f

/* Selects how the spectrogram magnitude is scaled, to compare their STFT timing cost against each other */

#define STFT_LOG_MODE_LINEAR                      0
#define STFT_LOG_MODE_LOG10F                      1
#define STFT_LOG_MODE_VLOG                        2
#define STFT_LOG_MODE_LOG10_FIXED                 3

#define STFT_LOG_MODE                             STFT_LOG_MODE_LOG10_FIXED

/* 20/ln(10): converts arm_vlog_f32's natural log output to 20*log10(x) with a single multiply */

#define STFT_20_OVER_LN10                         8.68588963806f

/* round(log10(2) * 65536): Q16.16 fixed-point constant used to turn the fixed-point log2 approximation below into log10 */

#define STFT_LOG2_TO_LOG10_Q16_16                 19728

#if STFT_LOG_MODE == STFT_LOG_MODE_LOG10F
#define STFT_MODE_SUFFIX                          "LOG10F"
#elif STFT_LOG_MODE == STFT_LOG_MODE_VLOG
#define STFT_MODE_SUFFIX                          "VLOG"
#elif STFT_LOG_MODE == STFT_LOG_MODE_LOG10_FIXED
#define STFT_MODE_SUFFIX                          "LOG10FIXED"
#else
#define STFT_MODE_SUFFIX                          "LINEAR"
#endif

#if STFT_DEBUG_MODE

/* CSV just logs buffer_index/elapsed time per buffer, as in the earlier log10f/vlog/linear comparisons - lightweight, no re-flash-free Python inspection. BINARY additionally saves the raw audio and full spectrogram of every buffer (see plot_stft_data.py), at the cost of a much bigger file and a per-buffer copy into external SRAM. Specific to STFT_DEBUG_MODE's own debug-file output - TRIGGERED_RECORDING_MODE writes real WAV files instead, see runTriggeredRecording(). */

#define STFT_OUTPUT_MODE_CSV                      0
#define STFT_OUTPUT_MODE_BINARY                   1

#define STFT_OUTPUT_MODE                          STFT_OUTPUT_MODE_BINARY

#if STFT_OUTPUT_MODE == STFT_OUTPUT_MODE_BINARY
#define STFT_DATA_FILENAME                        "STFT_DATA_" STFT_MODE_SUFFIX ".BIN"
#else
#define STFT_DATA_FILENAME                        "STFT_TIMING_" STFT_MODE_SUFFIX ".CSV"
#endif

#endif

#if STFT_LOG_MODE == STFT_LOG_MODE_LOG10_FIXED || STFT_NORMALIZATION_MODE == STFT_NORMALIZATION_PCEN

/* Fast log10 approximation with no floating point operations beyond reinterpreting the input's bit pattern and converting the final fixed-point result back to a float for storage. Based on the classic IEEE-754 bit-hack: for a positive float32 x, its raw bit pattern i satisfies log2(x) ~= i / 2^23 - 127. Since (i / 2^23) * 2^16 == i >> 7, that gives log2(x) in Q16.16 fixed point (value = integer / 65536) using only a shift and a subtract - no multiplication or division. log10 is then obtained with a single Q16.16 fixed-point multiply by log10(2), instead of the more expensive libm log10f().

   The naive bias of 127 is exact only at powers of two and otherwise always undershoots (the true log2(1+m) curve for mantissa m in [0,1) is concave, so the linear m approximation stays below it), with a worst case of 0.0861 (in log2 units) at m = 1/ln(2) - 1. Using a bias of 127 - 0.0430356660279671 = 126.95696433397204 instead re-centres that error around zero, halving the worst case to 0.0430 - equivalent to roughly 0.26dB instead of 0.52dB on the final 20*log10 result, for no extra cost since it's still a single constant.

   Guarded on STFT_NORMALIZATION_PCEN too (not just STFT_LOG_MODE_LOG10_FIXED): stftFastPow()
   below (PCEN's own fractional-exponent pow) reuses stftFastLog2Q16_16() directly, regardless
   of which STFT_LOG_MODE is selected - STFT_LOG_MODE only chooses between log10 variants for
   the EMA/NONE dB path, which STFT_NORMALIZATION_PCEN bypasses entirely (see
   computeSpectrogram() below), so the two macros are genuinely independent here. */

#define STFT_LOG2_BIAS_Q16_16                    8320252  /* round(126.95696433397204 * 65536) */

static inline int32_t stftFastLog2Q16_16(float32_t x) {

    int32_t i;

    memcpy(&i, &x, sizeof(i));

    return (i >> 7) - STFT_LOG2_BIAS_Q16_16;

}

#if STFT_LOG_MODE == STFT_LOG_MODE_LOG10_FIXED

static inline float32_t stftFastLog10(float32_t x) {

    int32_t log2Fixed = stftFastLog2Q16_16(x);

    int32_t log10Fixed = (int32_t)(((int64_t)log2Fixed * STFT_LOG2_TO_LOG10_Q16_16) >> 16);

    return (float32_t)log10Fixed / 65536.0f;

}

#endif

#if STFT_NORMALIZATION_MODE == STFT_NORMALIZATION_PCEN

/* Fast x^a approximation for PCEN's own fractional, per-band-LEARNED exponents (alpha, r -
   see model/pcen.h) - stftFastLog10 above can't be reused for these the way it's reused for
   the EMA/NONE dB path, since that has one FIXED exponent (always base-10) baked into a single
   shared constant (STFT_LOG2_TO_LOG10_Q16_16), while PCEN needs a different exponent per bin,
   decided by training, not known at compile time.

   pow(x, a) = exp2(a * log2(x)): reuses stftFastLog2Q16_16()'s exact approximation (and its
   same bias) to get log2(x), scales by the exponent, then reconstructs a float from the
   scaled Q16.16 value via the SAME bit trick in reverse (bits = (log2Fixed + BIAS) << 7,
   i.e. algebraically undoing stftFastLog2Q16_16's own (i >> 7) - BIAS) - not a new/unproven
   numerical scheme, the exact same approximation quality already validated and deployed for
   stftFastLog10 (~0.26dB / ~3% worst case on a single log10 call). End-to-end (both pow()
   calls PCEN actually makes, compounded, against exact float64 PCEN on a real recording -
   see test-audiomoth-cnn's fast-PCEN validation): mean absolute error 0.05, p99 0.13, against
   PCEN's own ~0-14 output range - the approximation error is a small fraction of PCEN's own
   dynamic range, not a dominant source of noise. x must be > 0 (a genuine bit-pattern
   assumption of the trick, same precondition stftFastLog2Q16_16 already has) - every call site
   below adds a small positive floor before calling this, same convention as STFT_LOG_EPSILON
   in the dB path. */

static inline float32_t stftFastExp2Q16_16(int32_t log2Fixed) {

    int32_t i = (log2Fixed + STFT_LOG2_BIAS_Q16_16) << 7;

    float32_t result;

    memcpy(&result, &i, sizeof(i));

    return result;

}

static inline float32_t stftFastPow(float32_t x, float32_t a) {

    int32_t log2Fixed = stftFastLog2Q16_16(x);

    int32_t scaledLog2Fixed = (int32_t)((float32_t)log2Fixed * a);

    return stftFastExp2Q16_16(scaledLog2Fixed);

}

#endif

#endif

#if STFT_DEBUG_MODE

/* Number of records buffered between the ISR (producer) and the main loop (consumer, writes to SD) - STFT_DEBUG_MODE's own debug-file queue; TRIGGERED_RECORDING_MODE has its own separate triggerQueue further below. */

#define STFT_RECORD_QUEUE_LENGTH                  8

#pragma pack(push, 1)

#if STFT_OUTPUT_MODE == STFT_OUTPUT_MODE_BINARY

/* Written once at the start of the file so a Python script can parse it without hard-coding these constants */

typedef struct {
    uint32_t sampleRate;
    uint32_t audioLength;
    uint32_t fftLength;
    uint32_t hopLength;
    uint32_t numberOfFrames;
    uint32_t numberOfBins;
} stftFileHeader_t;

/* One record per buffer: the raw audio that produced the spectrogram, alongside the spectrogram itself and how long it took to compute */

typedef struct {
    uint32_t bufferIndex;
    uint32_t elapsedMicroseconds;
    int16_t audio[MAXIMUM_SAMPLES_IN_DMA_TRANSFER];
    float32_t spectrogram[STFT_NUMBER_OF_FRAMES][STFT_NUMBER_OF_BINS];
    float32_t modelLogit;
} stftRecord_t;

#else

/* CSV mode only ever needs the timing of each buffer */

typedef struct {
    uint32_t bufferIndex;
    uint32_t elapsedMicroseconds;
    uint32_t inferenceMicroseconds;
    ModelInferenceTiming_t modelTiming;
#if STFT_DEBUG_ATTENTION_TEST
    uint32_t attentionMicroseconds;
    /* Per-sub-stage breakdown - see AttnLayerTiming_t/ATTN_TIME_STAGE further below (defined
       after this struct, hence plain uint32_t fields here rather than embedding that type
       directly - this file's own STFT_NORMALIZATION_MODE/model-inference sections are
       similarly forward-referenced by name only where needed, not reordered, so keeping this
       consistent rather than moving AttnLayerTiming_t's definition up here instead). */
    uint32_t attnQkvMicroseconds;
    uint32_t attnCoreMicroseconds;
    uint32_t attnOutProjMicroseconds;
    uint32_t attnLayerNorm1Microseconds;
    uint32_t attnFfnMicroseconds;
    uint32_t attnLayerNorm2Microseconds;
#endif
} stftRecord_t;

#endif

#pragma pack(pop)

#endif

static arm_rfft_fast_instance_f32 stftFFTInstance;

static float32_t stftWindow[STFT_FFT_LENGTH];

static float32_t stftFrame[STFT_FFT_LENGTH];

static float32_t stftFFTOutput[STFT_FFT_LENGTH];

static float32_t stftMagnitude[STFT_NUMBER_OF_BINS];

/* Spectrogram of the buffer currently being processed. In BINARY output mode this is genuinely read (copied into a record below), but in CSV mode nothing ever reads it back - so it must stay volatile, otherwise the compiler proves the whole magnitude/log computation feeding it is dead and removes it, silently invalidating any timing measurement. */

static volatile float32_t stftSpectrogramDb[STFT_NUMBER_OF_FRAMES][STFT_NUMBER_OF_BINS];

#if STFT_DEBUG_MODE

#if STFT_OUTPUT_MODE == STFT_OUTPUT_MODE_BINARY

/* Record queue lives in external SRAM - each record (~5.5KB) is far too large for the 32KB of internal RAM to hold more than one or two */

static stftRecord_t *stftRecordQueue = (stftRecord_t*)AM_EXTERNAL_SRAM_START_ADDRESS;

#else

/* CSV records are tiny (8 bytes), internal RAM is fine */

static stftRecord_t stftRecordQueue[STFT_RECORD_QUEUE_LENGTH];

#endif

static volatile uint32_t stftRecordQueueWriteIndex;

static volatile uint32_t stftRecordQueueReadIndex;

static volatile uint32_t stftBufferCounter;

#endif

#if STFT_NORMALIZATION_MODE == STFT_NORMALIZATION_EMA

/* Persistent streaming RMS estimate (mean-square power, raw ADC scale) - see
   STFT_RISE_EMA_ALPHA/STFT_FALL_EMA_ALPHA's comment above. Reset once per recording (resetRMSNormalization(),
   called from runSTFTFeasibilityTest() or runTriggeredRecording() alongside
   ModelInference_reset()), then updated once
   per buffer by computeBufferGain() below - never reset mid-recording. */

static float stftRunningMeanSq;

#endif

#if STFT_NORMALIZATION_MODE == STFT_NORMALIZATION_PCEN

/* Persistent PER-BAND running power estimate (PCEN's own "M", see model/pcen.h's
   model_pcen_ema_alpha and computeSpectrogram() below) - the PCEN equivalent of
   stftRunningMeanSq above, one value per frequency bin instead of a single global scalar,
   since PCEN's whole point is that a bin's own recent history can differ from the buffer's
   overall mean-square (see main.c's STFT_NORMALIZATION_MODE comment). Reset once per
   recording (resetRMSNormalization() below), then updated once per FRAME (not per buffer -
   PCEN runs on the post-FFT per-bin magnitude, at STFT_NUMBER_OF_FRAMES-per-buffer
   resolution, not the waveform-domain per-buffer resolution stftRunningMeanSq uses) by
   computeSpectrogram() - never reset mid-recording. */

static float pcenRunningM[STFT_NUMBER_OF_BINS];

#endif

#if STFT_NORMALIZATION_MODE == STFT_NORMALIZATION_EMA

static void resetRMSNormalization(void) {

    float targetRmsRaw = STFT_TARGET_RMS * STFT_INT16_FULL_SCALE;

    stftRunningMeanSq = targetRmsRaw * targetRmsRaw;

}

/* Updates the running mean-square estimate with this buffer's own power and returns the
   linear gain to apply to its samples - matches dataset.streaming_rms_normalize()'s
   per-buffer loop body exactly (mean-square accumulate, EMA update, gain = target/sqrt(...)).
   +1.0f in the sqrt matches that function's eps=1.0 (avoids dividing by zero during true
   digital silence, negligible next to any real signal at this raw int16 scale). */

static float computeBufferGain(const int16_t *source) {

    float targetRmsRaw = STFT_TARGET_RMS * STFT_INT16_FULL_SCALE;

    float sum = 0.0f;

    for (uint32_t i = 0; i < MAXIMUM_SAMPLES_IN_DMA_TRANSFER; i += 1) {

        float sample = (float32_t)source[i];

        sum += sample * sample;

    }

    float bufferMeanSq = sum / (float)MAXIMUM_SAMPLES_IN_DMA_TRANSFER;

    float alpha = (bufferMeanSq > stftRunningMeanSq) ? STFT_RISE_EMA_ALPHA : STFT_FALL_EMA_ALPHA;

    stftRunningMeanSq = alpha * bufferMeanSq + (1.0f - alpha) * stftRunningMeanSq;

    return targetRmsRaw / sqrtf(stftRunningMeanSq + 1.0f);

}

#else

/* STFT_NORMALIZATION_NONE and STFT_NORMALIZATION_PCEN both apply no waveform-domain gain at
   all (compiled out entirely rather than short-circuited, same reasoning as before: also
   measures/saves the mean-square accumulate loop's own compute cost, not just its effect on
   the model's input) - PCEN's own per-band adaptation happens AFTER the FFT, on the
   magnitude, in computeSpectrogram() below, not here. */

static void resetRMSNormalization(void) {

#if STFT_NORMALIZATION_MODE == STFT_NORMALIZATION_PCEN

    for (uint32_t bin = 0; bin < STFT_NUMBER_OF_BINS; bin += 1) {

        pcenRunningM[bin] = model_pcen_initial_m[bin];

    }

#endif

}

static float computeBufferGain(const int16_t *source) {

    (void)source;

    return 1.0f;

}

#endif

#if STFT_NORMALIZATION_MODE == STFT_NORMALIZATION_PCEN

/* One bin's PCEN value for the current frame, updating its persistent pcenRunningM[bin] in
   place first - matches student_model._pcen()'s per-frame loop body exactly:
   M[t] = (1-ema_alpha)*M[t-1] + ema_alpha*E[t], output = (E/(M+eps)^alpha + delta)^r - delta^r
   (E = power = magnitude^2, see model/pcen.h's own docstring for why power not amplitude).
   Factored out of computeSpectrogram() below since it's called from 3 different places there
   (bin 0, the Nyquist bin, and the main per-bin loop - see that function's own comment on why
   those are separate call sites, not merely a stylistic choice) - inlining the same 6-line
   formula 3 times would risk exactly the kind of copy-paste drift this codebase's "one shared
   implementation" convention (e.g. run_feature_stack_s8 on the Python side) exists to avoid.

   Never called with magnitude/power == 0 in a way that matters: for the 3 masked bins
   (bin < STFT_HIGHPASS_BIN), power is always exactly 0, so ratio is always exactly 0 and the
   result is always exactly delta^r - delta^r == 0 REGARDLESS of pcenRunningM's value - matching
   every other STFT_LOG_MODE's own masked-bin handling (a fixed, magnitude-independent output),
   so no bin-range special-casing is needed here, only in the caller's masking of `magnitude`
   itself (same as every other mode already does). delta[bin] > 0 always (softplus range) keeps
   `base` (the argument to the final stftFastPow) strictly positive even when ratio is exactly
   0, satisfying stftFastPow's own x>0 precondition unconditionally - no epsilon-flooring needed
   here the way STFT_LOG_EPSILON floors the dB path's log argument. */

static inline float32_t computePcenBin(uint32_t bin, float32_t magnitude) {

    float32_t power = magnitude * magnitude;

    pcenRunningM[bin] = model_pcen_ema_alpha[bin] * power + (1.0f - model_pcen_ema_alpha[bin]) * pcenRunningM[bin];

    float32_t denom = stftFastPow(pcenRunningM[bin] + MODEL_PCEN_EPS, model_pcen_alpha[bin]);

    float32_t ratio = power / denom;

    float32_t base = ratio + model_pcen_delta[bin];

    return stftFastPow(base, model_pcen_r[bin]) - model_pcen_delta_pow_r[bin];

}

#endif

/* Computes the RMS-normalized, high-pass-masked log-magnitude (or, in STFT_NORMALIZATION_PCEN,
   PCEN-compressed) STFT of one 64ms buffer into
   the shared stftSpectrogramDb - shared by STFT_DEBUG_MODE's ISR (timed separately from
   ModelInference_run() there, for the existing per-stage CSV/BINARY output) and
   TRIGGERED_RECORDING_MODE's ISR (which only needs the resulting logit, no per-stage
   timing). Factored out unchanged from what was previously inlined directly in the
   STFT-debug ISR - no behaviour change for that mode. */

static void computeSpectrogram(const int16_t *source) {

    float bufferGain = computeBufferGain(source);

    for (uint32_t frame = 0; frame < STFT_NUMBER_OF_FRAMES; frame += 1) {

        uint32_t offset = frame * STFT_HOP_LENGTH;

        for (uint32_t i = 0; i < STFT_FFT_LENGTH; i += 1) {

            stftFrame[i] = (float32_t)source[offset + i] * stftWindow[i] * bufferGain;

        }

        arm_rfft_fast_f32(&stftFFTInstance, stftFrame, stftFFTOutput, 0);

        /* CMSIS-DSP packs the real DC and Nyquist bins into the first complex slot of the output */

#if STFT_NORMALIZATION_MODE == STFT_NORMALIZATION_PCEN

        /* Bypasses STFT_LOG_MODE entirely - PCEN replaces the whole dB conversion, not just
           one of its variants (matching student_model.py's LogSpectrogram.forward(): in
           normalization="pcen" mode, self.pcen(magnitude) is returned directly, _fast_log10_db
           is never called at all). Bin 0 (DC) forced to 0 the same way every other mode forces
           it to a fixed, magnitude-independent floor - see computePcenBin's own docstring for
           why that's exactly right for PCEN too, not merely a convenient approximation. */

        stftSpectrogramDb[frame][0] = computePcenBin(0, 0.0f);

        stftSpectrogramDb[frame][STFT_FFT_LENGTH / 2] = computePcenBin(STFT_FFT_LENGTH / 2, fabsf(stftFFTOutput[1]));

        for (uint32_t bin = 1; bin < STFT_FFT_LENGTH / 2; bin += 1) {

            float32_t re = stftFFTOutput[2 * bin];

            float32_t im = stftFFTOutput[2 * bin + 1];

            float32_t magnitude = (bin < STFT_HIGHPASS_BIN) ? 0.0f : sqrtf(re * re + im * im);

            stftSpectrogramDb[frame][bin] = computePcenBin(bin, magnitude);

        }

#elif STFT_LOG_MODE == STFT_LOG_MODE_LOG10F

        /* Bin 0 (DC, 0Hz) is always below STFT_HIGHPASS_BIN - masked to the epsilon floor, matching student_model.py's LogSpectrogram exactly (magnitude zeroed before the log, not the log output clamped after) */

        stftSpectrogramDb[frame][0] = 20.0f * log10f(STFT_LOG_EPSILON);

        stftSpectrogramDb[frame][STFT_FFT_LENGTH / 2] = 20.0f * log10f(fabsf(stftFFTOutput[1]) + STFT_LOG_EPSILON);

        for (uint32_t bin = 1; bin < STFT_FFT_LENGTH / 2; bin += 1) {

            float32_t re = stftFFTOutput[2 * bin];

            float32_t im = stftFFTOutput[2 * bin + 1];

            float32_t magnitude = (bin < STFT_HIGHPASS_BIN) ? 0.0f : sqrtf(re * re + im * im);

            stftSpectrogramDb[frame][bin] = 20.0f * log10f(magnitude + STFT_LOG_EPSILON);

        }

#elif STFT_LOG_MODE == STFT_LOG_MODE_VLOG

        stftMagnitude[0] = STFT_LOG_EPSILON;

        stftMagnitude[STFT_FFT_LENGTH / 2] = fabsf(stftFFTOutput[1]) + STFT_LOG_EPSILON;

        for (uint32_t bin = 1; bin < STFT_FFT_LENGTH / 2; bin += 1) {

            float32_t re = stftFFTOutput[2 * bin];

            float32_t im = stftFFTOutput[2 * bin + 1];

            float32_t magnitude = (bin < STFT_HIGHPASS_BIN) ? 0.0f : sqrtf(re * re + im * im);

            stftMagnitude[bin] = magnitude + STFT_LOG_EPSILON;

        }

        /* arm_vlog_f32 computes the natural log of the whole vector in one call; converted to 20*log10(x) with a single multiply afterwards */

        arm_vlog_f32(stftMagnitude, stftMagnitude, STFT_NUMBER_OF_BINS);

        for (uint32_t bin = 0; bin < STFT_NUMBER_OF_BINS; bin += 1) {

            stftSpectrogramDb[frame][bin] = STFT_20_OVER_LN10 * stftMagnitude[bin];

        }

#elif STFT_LOG_MODE == STFT_LOG_MODE_LOG10_FIXED

        stftSpectrogramDb[frame][0] = 20.0f * stftFastLog10(STFT_LOG_EPSILON);

        stftSpectrogramDb[frame][STFT_FFT_LENGTH / 2] = 20.0f * stftFastLog10(fabsf(stftFFTOutput[1]) + STFT_LOG_EPSILON);

        for (uint32_t bin = 1; bin < STFT_FFT_LENGTH / 2; bin += 1) {

            float32_t re = stftFFTOutput[2 * bin];

            float32_t im = stftFFTOutput[2 * bin + 1];

            float32_t magnitude = (bin < STFT_HIGHPASS_BIN) ? 0.0f : sqrtf(re * re + im * im);

            stftSpectrogramDb[frame][bin] = 20.0f * stftFastLog10(magnitude + STFT_LOG_EPSILON);

        }

#else

        stftSpectrogramDb[frame][0] = fabsf(stftFFTOutput[0]);

        stftSpectrogramDb[frame][STFT_FFT_LENGTH / 2] = fabsf(stftFFTOutput[1]);

        for (uint32_t bin = 1; bin < STFT_FFT_LENGTH / 2; bin += 1) {

            float32_t re = stftFFTOutput[2 * bin];

            float32_t im = stftFFTOutput[2 * bin + 1];

            stftSpectrogramDb[frame][bin] = sqrtf(re * re + im * im);

        }

#endif

    }

}

#if STFT_DEBUG_MODE && STFT_DEBUG_ATTENTION_TEST

/* Isolated ALiBi windowed-attention encoder-layer timing test - see this section's own
   comment on STFT_DEBUG_ATTENTION_TEST above for the full scope/purpose (a compute-cost
   probe, not a step towards real deployment: dummy weights, dummy input, no real ViT export
   exists). Ports test-audiomoth-cnn/student_model.py's _AlibiEncoderLayer/
   _AlibiSelfAttention - one encoder layer; that checkpoint uses depth=2, so double this
   measurement (plus patch_embed/the classifier, both small relative to a layer - see
   EXPERIMENTS.md's own parameter breakdown) for a full-model estimate.

   ATTN_EMBED_DIM is deliberately a knob, not hardwired to one checkpoint's shape: the first
   real measurement (embed_dim=64, n_heads=4, context_size=32, matching
   student_cnn_vit_alibi_pcen.pt, the best-performing checkpoint at the time) came back at
   ~209ms for a single layer's worth of buffer, ~3x the entire 64ms budget on its own (see
   EXPERIMENTS.md) - set to 32 to measure the smaller embed_dim currently being trained
   (--vit-embed-dim 32) as a candidate for closing that gap. Every other #define below derives
   from it, so this is the only line that needs to change to compare configurations.

   Symmetric (zero-point-free) INT8 quantization throughout - input_offset/filter_offset/
   output_offset all 0 - the same convention model_inference.c's own conv/LSTM kernels
   already use (their own export comments note "input_zero_point is always 0 here"), so this
   stays consistent with how a real ViT export would eventually look rather than inventing a
   different convention just for this probe. */

#define ATTN_EMBED_DIM                            32

#define ATTN_N_HEADS                              4

#define ATTN_HEAD_DIM                              (ATTN_EMBED_DIM / ATTN_N_HEADS)

#define ATTN_CONTEXT_SIZE                         32

#define ATTN_FFN_DIM                               (4 * ATTN_EMBED_DIM)

/* The real checkpoint (student_cnn_vit_alibi_pcen.pt) has depth=2, not 1 - the single-layer
   measurements so far understated the real stacked cost. attnRunLayer() below now takes a
   layer index so each of the ATTN_N_LAYERS layers gets its own K/V ring buffer (a real 2-layer
   model's layer 2 attends over layer 2's own history, not layer 1's) - see attnKWindow/
   attnVWindowT's own comment for why the WEIGHTS are still shared dummy data across layers
   rather than duplicated: it doesn't change the timing at all (CMSIS-NN/attnVecMatMultS8's
   execution time depends only on shapes, never on the values read), so duplicating another
   ~12KB of flash for a second identical-shaped dummy weight set would add flash usage without
   adding any timing information this probe cares about. */

#define ATTN_N_LAYERS                              2

/* Dummy multiplier/shift shared by every requantizing call below - not calibrated against
   any real activation range (there is none, see above), just a fixed, bounded scale (~0.5,
   Q31 convention) that keeps int8 outputs from every stage saturating identically on every
   call - enough for a compute-cost probe, not remotely enough for a real accuracy claim. */

#define ATTN_DUMMY_MULTIPLIER                     1073741824

#define ATTN_DUMMY_SHIFT                          -1

/* Weights/biases below used to be filled at runtime (attnFillDummyInt8(), a simple
   deterministic pattern, never trained/exported - see this section's own header comment) into
   writable arrays living in external SRAM, because non-const arrays can't have compile-time
   initializers and the ~13KB total (at this ATTN_EMBED_DIM) didn't fit the remaining internal
   RAM headroom. Third experiment in this series (after idea #1/softmax-batching, idea #2/
   hand-written dot product, and moving K/V to internal RAM - see that comment above): the
   qkv/out_proj/FFN stages' own weight reads were the one remaining external-SRAM access this
   probe hadn't isolated, and a REAL exported ViT would have these as `const` arrays living in
   FLASH instead, exactly like model.h/pcen.h's own trained weights - flash reads are already
   proven fast enough for this hardware's real-time budget by the deployed LSTM's own model.c.
   So: same deterministic values (bit-for-bit, see scripts/gen_attn_dummy_weights.py), but now
   generated at build time into src/attn_dummy_weights.h as `const` arrays, which the linker
   places in flash (.rodata) instead of internal RAM or external SRAM - isolating "where do the
   weights live" as the sole remaining variable versus the previous external-SRAM measurement.
   #include lives right below; attnQkvWeight etc. now just alias the generated *Const names. */

#include "attn_dummy_weights.h"

#define attnQkvWeight                             attnQkvWeightConst
#define attnQkvBias                               attnQkvBiasConst
#define attnOutProjWeight                         attnOutProjWeightConst
#define attnOutProjBias                           attnOutProjBiasConst
#define attnLinear1Weight                         attnLinear1WeightConst
#define attnLinear1Bias                           attnLinear1BiasConst
#define attnLinear2Weight                         attnLinear2WeightConst
#define attnLinear2Bias                           attnLinear2BiasConst

/* Per-layer, per-head ring buffers of the last ATTN_CONTEXT_SIZE frames' K/V. K is stored
   [context][head_dim] (natural layout, matches attnVecMatMultS8's rhs convention for computing
   Q . K^T); V is stored PRE-TRANSPOSED as [head_dim][context] so the exact same function
   computes attn . V without needing a second, differently-shaped kernel - see
   student_model.py's own _AlibiSelfAttention docstring for why V needs this layout on the
   training side too. Never reset mid-recording, like PCEN's/the LSTM's own persistent state -
   cold-start correctness at the very start of a recording, before the window has filled once,
   is out of scope here (see this section's own header comment).

   Layer 0's ring buffer lives in internal RAM (2KB) - the placement confirmed to cut attn_core
   by ~13% earlier in this session. Layer 1's doesn't also fit: internal RAM had only ~1KB of
   headroom left with layer 0's buffer already in it (confirmed directly - linking a second
   2KB buffer there overflowed RAM). So layer 1 falls back to external SRAM, the same tradeoff
   the WEIGHTS used to have before they moved to flash - this makes layer 1's attn_core
   somewhat pessimistic relative to layer 0's for this specific 2-layer measurement (by roughly
   that same ~13%). A real deployment sized for depth=2 would likely need a smaller
   ATTN_CONTEXT_SIZE or embed_dim to fit both layers' K/V in fast RAM simultaneously - a design
   tradeoff out of scope for this timing probe. Bare AM_EXTERNAL_SRAM_START_ADDRESS (not an
   offset like TRIGGER_QUEUE_BASE_OFFSET uses) - safe here because this probe has only ever been
   measured in CSV output mode, where stftRecordQueue stays in internal RAM and never claims
   this address (see stftRecordQueue's own STFT_OUTPUT_MODE_BINARY branch above); it would
   collide if this probe were ever run in BINARY mode instead, not attempted this session. */

static int8_t attnKWindowLayer0[ATTN_N_HEADS][ATTN_CONTEXT_SIZE][ATTN_HEAD_DIM];
static int8_t attnVWindowTLayer0[ATTN_N_HEADS][ATTN_HEAD_DIM][ATTN_CONTEXT_SIZE];

typedef struct {
    int8_t kWindow[ATTN_N_HEADS][ATTN_CONTEXT_SIZE][ATTN_HEAD_DIM];
    int8_t vWindowT[ATTN_N_HEADS][ATTN_HEAD_DIM][ATTN_CONTEXT_SIZE];
} attnExternalWindow_t;

static attnExternalWindow_t *attnWindowLayer1 = (attnExternalWindow_t*)AM_EXTERNAL_SRAM_START_ADDRESS;

/* Precomputed ALiBi bias, per head/window-offset - see student_model.py's TinyViT._alibi_bias
   for the reference (fixed geometric slopes, not learned - see _alibi_slopes there, so a real
   deployment could genuinely precompute this exact table, unlike PCEN's learned constants).
   Added to the QK^T scores after CMSIS-NN's own requantization (int8 domain) rather than to
   the pre-requantization int32 accumulator - a deliberate simplification acceptable for a
   compute-cost probe, same spirit as this whole section's dummy quantization scale. Small
   (ATTN_N_HEADS x ATTN_CONTEXT_SIZE bytes = 128B) - stays in internal RAM, unlike everything
   above. */

static int8_t attnAlibiBias[ATTN_N_HEADS][ATTN_CONTEXT_SIZE];

static uint32_t attnWindowPos[ATTN_N_LAYERS];

static bool attnStateInitialised;

/* Weight/bias filling is gone from here - attnQkvWeight etc. are now the const/flash arrays
   from attn_dummy_weights.h (see that #include above). What's left to initialise at runtime is
   just the ALiBi table and the K/V ring buffers, neither of which can be const (ALiBi could be,
   in principle, but is cheap enough to compute once here instead of also generating it; K/V is
   genuine runtime state, never fixed). Renamed from attnInitDummyWeights to reflect that it no
   longer touches any weights. */

static void attnInitRuntimeState(void) {

    uint32_t h, w;

    for (h = 0; h < ATTN_N_HEADS; h += 1) {

        for (w = 0; w < ATTN_CONTEXT_SIZE; w += 1) {

            /* Matches the shape (not the trained values) of a real ALiBi table: a per-head
               constant slope, linearly decaying with distance from the current position. */

            attnAlibiBias[h][w] = (int8_t)(-(int32_t)w / (int32_t)(h + 1));

        }

    }

    memset(attnKWindowLayer0, 0, sizeof(attnKWindowLayer0));

    memset(attnVWindowTLayer0, 0, sizeof(attnVWindowTLayer0));

    memset(attnWindowLayer1, 0, sizeof(attnExternalWindow_t));

    memset(attnWindowPos, 0, sizeof(attnWindowPos));

    attnStateInitialised = true;

}

/* LayerNorm has no CMSIS-NN int8 kernel (confirmed - none of BatchNorm's "fold into the
   preceding layer's weights at export time" trick applies here, since LayerNorm's mean/
   variance are computed per-token at runtime, not fixed after training) - done in float
   instead. Mean/variance are a plain hand-written reduction rather than CMSIS-DSP's own
   arm_mean_f32/arm_var_f32 - not a performance choice, a build one: CMSIS-DSP's
   StatisticsFunctions directory ships both the individual per-function .c files AND a
   StatisticsFunctions.c aggregator that #includes them all, and this project's Makefile adds
   every .c file in a source directory it's given - pulling in the whole directory linked both
   copies of every function and failed with "multiple definition" (confirmed directly). Same
   convention already used elsewhere in this file for small/simple reductions (e.g.
   runClassifier() in model_inference.c) rather than pulling in more CMSIS machinery for a
   64-element loop. arm_sqrt_f32 (a real hardware VSQRT instruction on this Cortex-M4F, not a
   bit-trick approximation like stftFastPow's - sqrt is one of the few transcendental-ish
   operations this FPU actually has natively) is still used for the standard deviation - it's
   header-only/inline (see arm_math.h), no separate source file to conflict.

   gamma/beta are omitted (identity) - this probe only needs a realistic AMOUNT of floating-
   point work per call, not trained affine parameters that don't exist yet anyway. Dequantizes
   with a fixed, arbitrary scale (1.0) and requantizes the same way - again, only the
   operation COUNT matters here. */

static void attnLayerNorm(int8_t *data, uint32_t n) {

    float32_t buffer[ATTN_EMBED_DIM];

    uint32_t i;

    float32_t sum = 0.0f;

    for (i = 0; i < n; i += 1) {

        buffer[i] = (float32_t)data[i];

        sum += buffer[i];

    }

    float32_t mean = sum / (float32_t)n;

    float32_t sumSquaredDiff = 0.0f;

    for (i = 0; i < n; i += 1) {

        float32_t diff = buffer[i] - mean;

        sumSquaredDiff += diff * diff;

    }

    float32_t variance = sumSquaredDiff / (float32_t)n;

    float32_t std;

    arm_sqrt_f32(variance + 1.0e-5f, &std);

    for (i = 0; i < n; i += 1) {

        float32_t normalised = (buffer[i] - mean) / std;

        int32_t quantised = (int32_t)normalised;

        data[i] = (int8_t)(quantised < -128 ? -128 : (quantised > 127 ? 127 : quantised));

    }

}

/* Per-sub-stage breakdown of attnRunLayer() below, accumulated (added to, not overwritten -
   see ATTN_TIME_STAGE) across every call within a buffer, exactly like attentionMicroseconds
   itself - so this stays directly comparable (same units, same "whole buffer" scope) to it
   and to the existing model_inference.c per-stage columns. Added after the very first real
   hardware measurement (a single aggregate attention_us) came back at ~200ms - more than 3x
   the entire 64ms buffer budget for ONE layer alone - to find out which specific step is
   actually responsible before drawing any conclusion about why, see EXPERIMENTS.md. */

typedef struct {
    uint32_t qkvMicroseconds;
    uint32_t attentionCoreMicroseconds;
    uint32_t outProjMicroseconds;
    uint32_t layerNorm1Microseconds;
    uint32_t ffnMicroseconds;
    uint32_t layerNorm2Microseconds;
} AttnLayerTiming_t;

#define ATTN_TIME_STAGE(field, call) { \
    uint32_t attnStageStartTime, attnStageStartUs, attnStageEndTime, attnStageEndUs; \
    AudioMoth_getTimeMicroseconds(&attnStageStartTime, &attnStageStartUs); \
    call; \
    AudioMoth_getTimeMicroseconds(&attnStageEndTime, &attnStageEndUs); \
    timing->field += (attnStageEndTime - attnStageStartTime) * MICROSECONDS_IN_SECOND + attnStageEndUs - attnStageStartUs; \
}

/* Hand-written replacement for arm_nn_vec_mat_mult_t_s8(), specialised for the one case this
   file ever calls it with: lhs_offset=0, rhs_offset=0, dst_offset=0, bias=NULL, address_offset=1
   (symmetric quantisation throughout, no bias tensor, contiguous dst). dst[r] = requantize(sum_c
   lhs[c]*rhs[r*rhs_cols+c]), clamped to int8 - bit-exact with what arm_nn_vec_mat_mult_t_s8 would
   compute for this same input (verified against its actual source, not assumed), since with both
   offsets at 0 its own lhs_offset/rhs_offset adjustment terms are no-ops and it calls the same
   header-inline arm_nn_requantize() with the same (multiplier, shift) this does. See the call
   sites in attnRunLayer() (idea #2) for why this may or may not be faster than the library call
   on real hardware - CMSIS-NN's own SMLAD-based DSP path is not a naive loop, so this is an
   experiment, not a guaranteed win. */

static inline void attnVecMatMultS8(const int8_t *lhs, const int8_t *rhs, int8_t *dst,
                                    int32_t multiplier, int32_t shift,
                                    uint32_t rhsCols, uint32_t rhsRows) {

    uint32_t r, c;

    for (r = 0; r < rhsRows; r += 1) {

        const int8_t *rhsRow = rhs + r * rhsCols;

        int32_t acc = 0;

        for (c = 0; c < rhsCols; c += 1) acc += (int32_t)lhs[c] * (int32_t)rhsRow[c];

        acc = arm_nn_requantize(acc, multiplier, shift);

        dst[r] = (int8_t)(acc < -128 ? -128 : (acc > 127 ? 127 : acc));

    }

}

/* One ALiBi windowed-attention encoder layer (self-attention + residual + LayerNorm, FFN +
   residual + LayerNorm - matches _AlibiEncoderLayer's own post-LN layout exactly, see its
   docstring) on a single raw-frame embedding. input/output: ATTN_EMBED_DIM int8 each (may
   alias the same buffer - output is only written at the very end, after every read of
   input has already happened). layer: which of the ATTN_N_LAYERS stacked layers this call is
   (see ATTN_N_LAYERS's own comment) - selects which layer's K/V ring buffer to read/update;
   the weights themselves are shared dummy data across layers (doesn't affect timing, see same
   comment). Updates that layer's persistent K/V ring buffer as a side effect (see its own
   comment above) - so repeated calls accumulate real windowed-attention history exactly like
   the deployed LSTM's own hidden state would, not independent from-scratch calls each time.
   timing: accumulated into (see ATTN_TIME_STAGE) - the caller zeroes it once per buffer, not
   once per call. */

static void attnRunLayer(const int8_t *input, int8_t *output, uint32_t layer, AttnLayerTiming_t *timing) {

    if (!attnStateInitialised) attnInitRuntimeState();

    /* layer 0's K/V ring buffer is in internal RAM, layer 1's is in external SRAM - see
       attnKWindowLayer0/attnWindowLayer1's own comment above for why. Selecting the pointer
       here (rather than a per-layer array of structs) keeps attnKWindow[h]/attnVWindowT[h]
       below reading identically regardless of which layer this call is for. */

    int8_t (*attnKWindow)[ATTN_CONTEXT_SIZE][ATTN_HEAD_DIM] = (layer == 0) ? attnKWindowLayer0 : attnWindowLayer1->kWindow;

    int8_t (*attnVWindowT)[ATTN_HEAD_DIM][ATTN_CONTEXT_SIZE] = (layer == 0) ? attnVWindowTLayer0 : attnWindowLayer1->vWindowT;

    cmsis_nn_context ctx = { NULL, 0 };

    cmsis_nn_activation activation = { -128, 127 };

    cmsis_nn_fc_params fcParams = { 0, 0, 0, activation };

    cmsis_nn_per_tensor_quant_params quantParams = { ATTN_DUMMY_MULTIPLIER, ATTN_DUMMY_SHIFT };

    cmsis_nn_dims inputDims = { 1, 1, 1, ATTN_EMBED_DIM };

    /* --- qkv projection (ATTN_EMBED_DIM -> 3*ATTN_EMBED_DIM) --- */

    int8_t qkv[3 * ATTN_EMBED_DIM];

    cmsis_nn_dims qkvFilterDims = { 3 * ATTN_EMBED_DIM, 1, 1, ATTN_EMBED_DIM };

    cmsis_nn_dims qkvBiasDims = { 1, 1, 1, 3 * ATTN_EMBED_DIM };

    cmsis_nn_dims qkvOutputDims = { 1, 1, 1, 3 * ATTN_EMBED_DIM };

    ATTN_TIME_STAGE(qkvMicroseconds,
        arm_fully_connected_s8(&ctx, &fcParams, &quantParams, &inputDims, input, &qkvFilterDims, attnQkvWeight,
                               &qkvBiasDims, attnQkvBias, &qkvOutputDims, qkv))

    int8_t *q = qkv;

    int8_t *k = qkv + ATTN_EMBED_DIM;

    int8_t *v = qkv + 2 * ATTN_EMBED_DIM;

    /* --- append this frame's K/V to the ring buffers, advance the write position --- */

    uint32_t h, i;

    for (h = 0; h < ATTN_N_HEADS; h += 1) {

        for (i = 0; i < ATTN_HEAD_DIM; i += 1) {

            attnKWindow[h][attnWindowPos[layer]][i] = k[h * ATTN_HEAD_DIM + i];

            attnVWindowT[h][i][attnWindowPos[layer]] = v[h * ATTN_HEAD_DIM + i];

        }

    }

    attnWindowPos[layer] = (attnWindowPos[layer] + 1) % ATTN_CONTEXT_SIZE;

    /* --- per-head windowed attention: Q.K^T (+ALiBi), softmax, attn.V --- */

    int8_t attnOut[ATTN_EMBED_DIM];

    /* Speedup idea #2 (see this session's own design discussion, EXPERIMENTS.md): replace the
       8 per-layer arm_nn_vec_mat_mult_t_s8 calls (4 heads x (Q.K^T, attn.V)) with a hand-written,
       inlined dot-product loop, attnVecMatMultS8() (defined above attnRunLayer). Idea #1 (batching
       all heads' softmax into one arm_softmax_s8 call) was tried first and measured on real
       hardware to save only ~0.5% of attn_core_us - reverted back to one arm_softmax_s8 call per
       head below (per-head scores[]/attnWeights[], not the shared [ATTN_N_HEADS][...] arrays idea
       #1 used).

       IMPORTANT CAVEAT (found by reading arm_nn_vec_mat_mult_t_s8.c's actual source in full before
       writing this, not assumed): on this target (Cortex-M4, ARM_MATH_DSP defined, no MVE), the
       rhs_offset==0 path we hit (our quantization is symmetric, rhs_offset is always 0) is NOT a
       naive scalar loop - it already uses SMLAD-based paired 16-bit MACs and processes 2 output
       rows per iteration. So attnVecMatMultS8()'s plain scalar loop is doing MORE arithmetic
       instructions per element than the library call, not fewer. What it removes instead is: the
       15-argument call's stack-marshaled-argument setup (paid once per call), the unconditional
       lhs_offset SXTAB16 adjustment the library applies even though our lhs_offset is always 0,
       and the row-pairing/remainder-handling bookkeeping paid once per row regardless of how few
       columns there are (which is the likely reason attn_core barely shrank when embed_dim halved:
       ATTN_CONTEXT_SIZE, i.e. the row count for Q.K^T, stayed fixed at 32, so that per-row
       bookkeeping cost didn't shrink with embed_dim even though the column-wise MAC work did).
       Net effect on real hardware is genuinely uncertain either direction - unlike idea #1, this
       is flagged upfront as an experiment, not a confident prediction. */

    int8_t scores[ATTN_CONTEXT_SIZE];

    int8_t attnWeights[ATTN_CONTEXT_SIZE];

    ATTN_TIME_STAGE(attentionCoreMicroseconds,
    {
        for (h = 0; h < ATTN_N_HEADS; h += 1) {

            attnVecMatMultS8(q + h * ATTN_HEAD_DIM, (const int8_t *)attnKWindow[h], scores,
                             ATTN_DUMMY_MULTIPLIER, ATTN_DUMMY_SHIFT, ATTN_HEAD_DIM, ATTN_CONTEXT_SIZE);

            for (i = 0; i < ATTN_CONTEXT_SIZE; i += 1) {

                int32_t biased = (int32_t)scores[i] + (int32_t)attnAlibiBias[h][i];

                scores[i] = (int8_t)(biased < -128 ? -128 : (biased > 127 ? 127 : biased));

            }

            arm_softmax_s8((const int8_t *)scores, 1, ATTN_CONTEXT_SIZE, ATTN_DUMMY_MULTIPLIER, ATTN_DUMMY_SHIFT,
                           -255, attnWeights);

            attnVecMatMultS8(attnWeights, (const int8_t *)attnVWindowT[h], attnOut + h * ATTN_HEAD_DIM,
                             ATTN_DUMMY_MULTIPLIER, ATTN_DUMMY_SHIFT, ATTN_CONTEXT_SIZE, ATTN_HEAD_DIM);

        }
    })

    /* --- output projection, residual, LayerNorm --- */

    int8_t projOut[ATTN_EMBED_DIM];

    cmsis_nn_dims squareFilterDims = { ATTN_EMBED_DIM, 1, 1, ATTN_EMBED_DIM };

    cmsis_nn_dims squareBiasDims = { 1, 1, 1, ATTN_EMBED_DIM };

    cmsis_nn_dims squareOutputDims = { 1, 1, 1, ATTN_EMBED_DIM };

    ATTN_TIME_STAGE(outProjMicroseconds,
        arm_fully_connected_s8(&ctx, &fcParams, &quantParams, &inputDims, attnOut, &squareFilterDims, attnOutProjWeight,
                               &squareBiasDims, attnOutProjBias, &squareOutputDims, projOut))

    int8_t hidden[ATTN_EMBED_DIM];

    for (i = 0; i < ATTN_EMBED_DIM; i += 1) {

        int32_t sum = (int32_t)projOut[i] + (int32_t)input[i];

        hidden[i] = (int8_t)(sum < -128 ? -128 : (sum > 127 ? 127 : sum));

    }

    ATTN_TIME_STAGE(layerNorm1Microseconds, attnLayerNorm(hidden, ATTN_EMBED_DIM))

    /* --- feedforward: linear1 -> ReLU -> linear2, residual, LayerNorm --- */

    int8_t ffnHidden[ATTN_FFN_DIM];

    cmsis_nn_dims ffn1FilterDims = { ATTN_FFN_DIM, 1, 1, ATTN_EMBED_DIM };

    cmsis_nn_dims ffn1BiasDims = { 1, 1, 1, ATTN_FFN_DIM };

    cmsis_nn_dims ffn1OutputDims = { 1, 1, 1, ATTN_FFN_DIM };

    int8_t ffnOut[ATTN_EMBED_DIM];

    cmsis_nn_dims ffnInputDims = { 1, 1, 1, ATTN_FFN_DIM };

    ATTN_TIME_STAGE(ffnMicroseconds,
    {
        arm_fully_connected_s8(&ctx, &fcParams, &quantParams, &inputDims, hidden, &ffn1FilterDims, attnLinear1Weight,
                               &ffn1BiasDims, attnLinear1Bias, &ffn1OutputDims, ffnHidden);

        arm_relu_q7(ffnHidden, ATTN_FFN_DIM);

        arm_fully_connected_s8(&ctx, &fcParams, &quantParams, &ffnInputDims, ffnHidden, &squareFilterDims, attnLinear2Weight,
                               &squareBiasDims, attnLinear2Bias, &squareOutputDims, ffnOut);
    })

    for (i = 0; i < ATTN_EMBED_DIM; i += 1) {

        int32_t sum = (int32_t)ffnOut[i] + (int32_t)hidden[i];

        output[i] = (int8_t)(sum < -128 ? -128 : (sum > 127 ? 127 : sum));

    }

    ATTN_TIME_STAGE(layerNorm2Microseconds, attnLayerNorm(output, ATTN_EMBED_DIM))

}

#endif

#if STFT_DEBUG_MODE

/* Replaces the normal recording ISR: instead of filtering the buffer into the SD card recording pipeline, compute its STFT and time how long that takes */

inline void AudioMoth_handleDirectMemoryAccessInterrupt(bool isPrimaryBuffer, int16_t **nextBuffer) {

    int16_t *source = isPrimaryBuffer ? primaryBuffer : secondaryBuffer;

    uint32_t startTime, startMicroseconds;

    AudioMoth_getTimeMicroseconds(&startTime, &startMicroseconds);

    computeSpectrogram(source);

    uint32_t endTime, endMicroseconds;

    AudioMoth_getTimeMicroseconds(&endTime, &endMicroseconds);

    uint32_t elapsedMicroseconds = (endTime - startTime) * MICROSECONDS_IN_SECOND + endMicroseconds - startMicroseconds;

    /* Runs the exported classifier on the buffer just computed, timed separately from the STFT above so elapsedMicroseconds stays comparable with every earlier STFT-only measurement. Still in the ISR, like the STFT itself. Runs in both output modes now: CSV keeps the per-stage timing, BINARY keeps the logit so it can be captured alongside the spectrogram that produced it and replayed through int8_sim.py on a PC to check the on-device CMSIS-NN execution against a bit-exact Python reference of the same int8 arithmetic. */

    uint32_t inferenceStartTime, inferenceStartMicroseconds;

    AudioMoth_getTimeMicroseconds(&inferenceStartTime, &inferenceStartMicroseconds);

    ModelInferenceTiming_t modelTiming;

    float modelLogit = ModelInference_run(stftSpectrogramDb, &modelTiming);

    uint32_t inferenceEndTime, inferenceEndMicroseconds;

    AudioMoth_getTimeMicroseconds(&inferenceEndTime, &inferenceEndMicroseconds);

    uint32_t inferenceMicroseconds = (inferenceEndTime - inferenceStartTime) * MICROSECONDS_IN_SECOND + inferenceEndMicroseconds - inferenceStartMicroseconds;

#if STFT_DEBUG_ATTENTION_TEST

    /* Isolated ALiBi attention-layer compute-cost probe (see attnRunLayer()'s own comment
       above, and STFT_DEBUG_ATTENTION_TEST's) - timed separately, alongside the real
       STFT/LSTM timing above, not instead of it: this never touches stftSpectrogramDb,
       ModelInference_run(), or anything the real classifier reads, so the real LSTM timing
       and logit above stay exactly what they always were. Runs the full ATTN_N_LAYERS-deep
       stack STFT_NUMBER_OF_FRAMES times, matching TinyViT's own raw-STFT-frame resolution (one
       set of calls per raw frame, not one per buffer - see student_model.py's frame_logits()/
       adaptive_avg_pool1d and this section's own header comment on why). Each layer's output
       feeds the next layer's input (attnLayerBuffer), and each frame's final (layer-N) output
       feeds back as the NEXT frame's layer-0 input (a real, if dummy-weighted, continuous
       stream of frame embeddings, not the same static input every time) - the K/V ring
       buffers persist across buffers exactly like they would in real continuous deployment.
       Real checkpoint depth is ATTN_N_LAYERS=2 (see its own comment) - single-layer numbers
       measured before this understated the real stacked cost. */

    static int8_t attnDummyEmbedding[ATTN_EMBED_DIM];

    uint32_t attentionStartTime, attentionStartMicroseconds;

    AudioMoth_getTimeMicroseconds(&attentionStartTime, &attentionStartMicroseconds);

    AttnLayerTiming_t attnTiming = { 0, 0, 0, 0, 0, 0 };

    uint32_t attnFrame, attnLayer;

    for (attnFrame = 0; attnFrame < STFT_NUMBER_OF_FRAMES; attnFrame += 1) {

        for (attnLayer = 0; attnLayer < ATTN_N_LAYERS; attnLayer += 1) {

            attnRunLayer(attnDummyEmbedding, attnDummyEmbedding, attnLayer, &attnTiming);

        }

    }

    uint32_t attentionEndTime, attentionEndMicroseconds;

    AudioMoth_getTimeMicroseconds(&attentionEndTime, &attentionEndMicroseconds);

    uint32_t attentionMicroseconds = (attentionEndTime - attentionStartTime) * MICROSECONDS_IN_SECOND + attentionEndMicroseconds - attentionStartMicroseconds;

#endif

    /* In binary mode, the raw audio and the spectrogram just computed are also copied into the next free record. That copy (a few KB to external SRAM) happens after elapsedMicroseconds is captured, so that value still reflects only the STFT computation, comparable with earlier measurements. */

    uint32_t nextWriteIndex = (stftRecordQueueWriteIndex + 1) & (STFT_RECORD_QUEUE_LENGTH - 1);

    if (nextWriteIndex != stftRecordQueueReadIndex) {

        stftRecordQueue[stftRecordQueueWriteIndex].bufferIndex = stftBufferCounter;

        stftRecordQueue[stftRecordQueueWriteIndex].elapsedMicroseconds = elapsedMicroseconds;

#if STFT_OUTPUT_MODE == STFT_OUTPUT_MODE_BINARY

        memcpy(stftRecordQueue[stftRecordQueueWriteIndex].audio, source, sizeof(stftRecordQueue[0].audio));

        memcpy(stftRecordQueue[stftRecordQueueWriteIndex].spectrogram, stftSpectrogramDb, sizeof(stftRecordQueue[0].spectrogram));

        stftRecordQueue[stftRecordQueueWriteIndex].modelLogit = modelLogit;

#else

        stftRecordQueue[stftRecordQueueWriteIndex].inferenceMicroseconds = inferenceMicroseconds;

        stftRecordQueue[stftRecordQueueWriteIndex].modelTiming = modelTiming;

#if STFT_DEBUG_ATTENTION_TEST

        stftRecordQueue[stftRecordQueueWriteIndex].attentionMicroseconds = attentionMicroseconds;

        stftRecordQueue[stftRecordQueueWriteIndex].attnQkvMicroseconds = attnTiming.qkvMicroseconds;

        stftRecordQueue[stftRecordQueueWriteIndex].attnCoreMicroseconds = attnTiming.attentionCoreMicroseconds;

        stftRecordQueue[stftRecordQueueWriteIndex].attnOutProjMicroseconds = attnTiming.outProjMicroseconds;

        stftRecordQueue[stftRecordQueueWriteIndex].attnLayerNorm1Microseconds = attnTiming.layerNorm1Microseconds;

        stftRecordQueue[stftRecordQueueWriteIndex].attnFfnMicroseconds = attnTiming.ffnMicroseconds;

        stftRecordQueue[stftRecordQueueWriteIndex].attnLayerNorm2Microseconds = attnTiming.layerNorm2Microseconds;

#endif

#endif

        stftRecordQueueWriteIndex = nextWriteIndex;

    }

    stftBufferCounter += 1;

}

/* Standalone debug entry point: sample the microphone at a raw, undecimated STFT_SAMPLE_RATE and log the time taken to compute the STFT of each buffer to a text file on the SD card, instead of recording WAV audio */

static void runSTFTFeasibilityTest(void) {

    /* Precompute the Hann analysis window and initialise the FFT */

    for (uint32_t i = 0; i < STFT_FFT_LENGTH; i += 1) {

        stftWindow[i] = 0.5f - 0.5f * cosf(STFT_TWO_PI * (float32_t)i / (float32_t)(STFT_FFT_LENGTH - 1));

    }

    /* arm_rfft_fast_init_f32() is a generic dispatcher with a switch-case over every
       supported FFT length (16 to 4096) - since STFT_FFT_LENGTH is a compile-time constant
       here but the dispatcher takes a runtime argument, the compiler can't prove which case
       executes, so gc-sections can't strip the other sizes' twiddle/bit-reversal tables:
       confirmed via the linker map, ~69KB of otherwise-unreachable flash for sizes 512-4096
       (see EXPERIMENTS.md). Calling the 256-specific init directly removes that ambiguity -
       only the 256-point tables are referenced, everything else gets linked out. Tied to
       STFT_FFT_LENGTH==256 specifically - the #error below catches anyone changing that
       define without updating this call to match. */

#if STFT_FFT_LENGTH != 256
#error "arm_rfft_fast_init_256_f32() below is hard-coded for STFT_FFT_LENGTH==256 - use arm_rfft_fast_init_f32(&stftFFTInstance, STFT_FFT_LENGTH) instead if this changes (costs ~69KB of unused twiddle tables for the other sizes, see EXPERIMENTS.md)"
#endif

    arm_rfft_fast_init_256_f32(&stftFFTInstance);

    /* Wait for the SD card to be present and the log file to open successfully before doing anything else. Without this, a missing/removed SD card leaves the device in an undefined state that is indistinguishable from a hang and only recoverable by re-flashing. The red LED blinks once a second while waiting, and inserting the card lets it recover on its own. */

    bool sdCardReady = false;

    while (sdCardReady == false) {

        bool fileSystemEnabled = AudioMoth_enableFileSystem(AM_SD_CARD_NORMAL_SPEED);

        sdCardReady = fileSystemEnabled && AudioMoth_openFile(STFT_DATA_FILENAME);

        if (sdCardReady == false) {

            AudioMoth_setRedLED(true);

            AudioMoth_delay(100);

            AudioMoth_setRedLED(false);

            AudioMoth_delay(900);

        }

    }

    /* Enable the microphone at a raw, undecimated 16kHz (no software decimation filter) */

    AudioMoth_enableExternalSRAM();

    AudioMoth_ignoreExternalMicrophone(false);

    AudioMoth_enableMicrophone(AM_NORMAL_GAIN_RANGE, AM_GAIN_MEDIUM, 4, 16, 1);

    AudioMoth_initialiseDirectMemoryAccess(primaryBuffer, secondaryBuffer, MAXIMUM_SAMPLES_IN_DMA_TRANSFER);

#if STFT_OUTPUT_MODE == STFT_OUTPUT_MODE_BINARY

    /* Write the file header now that the file is confirmed open, so a Python script can parse the rest without hard-coding these constants */

    static const stftFileHeader_t stftFileHeader = {
        .sampleRate = STFT_SAMPLE_RATE,
        .audioLength = MAXIMUM_SAMPLES_IN_DMA_TRANSFER,
        .fftLength = STFT_FFT_LENGTH,
        .hopLength = STFT_HOP_LENGTH,
        .numberOfFrames = STFT_NUMBER_OF_FRAMES,
        .numberOfBins = STFT_NUMBER_OF_BINS
    };

    AudioMoth_writeToFile((void*)&stftFileHeader, sizeof(stftFileHeader));

#else

    /* 224 was enough for the original 11-column CSV (confirmed: max real row length well
       under that). STFT_DEBUG_ATTENTION_TEST's extra 7 columns push the HEADER row alone to
       230 bytes - over the old 224-byte buffer (confirmed by direct measurement, not just
       eyeballing after this file's build test - a real stack buffer overflow, since sprintf
       here has no bounds checking). 384 comfortably covers both the header and the widest
       plausible data row (18 uint32 fields, each up to 10 digits + a comma) with real margin
       to spare, not just enough to survive today's exact column count. */

    static char lineBuffer[384];

#if STFT_DEBUG_ATTENTION_TEST

    uint32_t lineLength = sprintf(lineBuffer, "buffer_index,stft_time_us,inference_time_us,stage0_us,stage1_dw_us,stage1_pw_us,stage2_dw_us,stage2_pw_us,avgpool_us,lstm_us,classifier_us,attention_us,attn_qkv_us,attn_core_us,attn_outproj_us,attn_ln1_us,attn_ffn_us,attn_ln2_us\r\n");

#else

    uint32_t lineLength = sprintf(lineBuffer, "buffer_index,stft_time_us,inference_time_us,stage0_us,stage1_dw_us,stage1_pw_us,stage2_dw_us,stage2_pw_us,avgpool_us,lstm_us,classifier_us\r\n");

#endif

    AudioMoth_writeToFile(lineBuffer, lineLength);

#endif

    AudioMoth_syncFile();

    /* Start sampling and continuously drain the record queue to the SD card */

    stftBufferCounter = 0;

    stftRecordQueueWriteIndex = 0;

    stftRecordQueueReadIndex = 0;

    ModelInference_reset();

    resetRMSNormalization();

    AudioMoth_startMicrophoneSamples(STFT_SAMPLE_RATE);

    while (true) {

        /* Re-checked every cycle (~64ms) rather than just once at start-up, so that flipping the switch to USB at any time - not only before power-up - still lets the device fall through to the normal USB handling below and be reflashed */

        if (AudioMoth_getSwitchPosition() == AM_SWITCH_USB) {

            AudioMoth_disableMicrophone();

            AudioMoth_closeFile();

            return;

        }

        while (stftRecordQueueReadIndex != stftRecordQueueWriteIndex) {

#if STFT_OUTPUT_MODE == STFT_OUTPUT_MODE_BINARY

            AudioMoth_writeToFile(&stftRecordQueue[stftRecordQueueReadIndex], sizeof(stftRecord_t));

#else

#if STFT_DEBUG_ATTENTION_TEST

            lineLength = sprintf(lineBuffer, "%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\r\n",
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].bufferIndex,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].elapsedMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].inferenceMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.stage0Microseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.stage1DepthwiseMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.stage1PointwiseMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.stage2DepthwiseMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.stage2PointwiseMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.avgPoolMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.lstmMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.classifierMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].attentionMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].attnQkvMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].attnCoreMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].attnOutProjMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].attnLayerNorm1Microseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].attnFfnMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].attnLayerNorm2Microseconds);

#else

            lineLength = sprintf(lineBuffer, "%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\r\n",
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].bufferIndex,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].elapsedMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].inferenceMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.stage0Microseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.stage1DepthwiseMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.stage1PointwiseMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.stage2DepthwiseMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.stage2PointwiseMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.avgPoolMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.lstmMicroseconds,
                (unsigned long)stftRecordQueue[stftRecordQueueReadIndex].modelTiming.classifierMicroseconds);

#endif

            AudioMoth_writeToFile(lineBuffer, lineLength);

#endif

            AudioMoth_syncFile();

            stftRecordQueueReadIndex = (stftRecordQueueReadIndex + 1) & (STFT_RECORD_QUEUE_LENGTH - 1);

        }

        AudioMoth_sleep();

    }

}

#elif TRIGGERED_RECORDING_MODE

/* Continuously listens and runs the classifier every 64ms buffer; whenever its output
   crosses TRIGGER_PROBABILITY_THRESHOLD, records a new WAV file (with TRIGGER_PREROLL_SECONDS
   of pre-roll audio), keeping it open while re-triggered (a sliding TRIGGER_TAIL_SECONDS
   countdown, reset on every re-trigger) and closing it once the countdown lapses - one
   distinct file per detection event, not one continuous file with gaps (unlike the existing
   amplitude/frequency trigger feature in makeRecording() below, which never splits files). */

/* Tune this directly - the probability threshold from the PR-curve data
   test-audiomoth-cnn/export_int8.py already produces - rather than triggerLogitThreshold
   (see below), which is derived from it once at recording start, not something to edit by
   hand: logf()/expf() are too expensive to call every buffer in the ISR, but computing the
   equivalent raw-logit threshold a single time and comparing the raw logit against that
   (monotonic with sigmoid, so equivalent) costs nothing per buffer. */

/* 0.6747 - precision-at-recall>=95% threshold from export_int8.py --long-clips's real INT8
   evaluation of student_windowed_lstm_lstm_16_32_64_h64_pcen.pt (100 calibration clips, 20
   test clips, --calibration-percentile 99.9: float32 96.7% frame AP, real INT8 96.0% - the
   default max-based calibration gave only 87.1% (a real, since-diagnosed issue: PCEN's rare
   "attack overshoot" transients set an abs_max ~100x its typical range, wasting int8
   resolution on values that occur in well under 0.1% of frames - see this session's own
   design discussion/export_int8.py's own calibrate() docstring). Re-tune whenever model.h/
   pcen.h are re-exported from a different checkpoint - this is per-checkpoint, not a fixed
   constant. */

#define TRIGGER_PROBABILITY_THRESHOLD              0.8f

/* logit(p) = ln(p/(1-p)), the inverse of sigmoid - see TRIGGER_PROBABILITY_THRESHOLD's
   comment above. Computed once in runTriggeredRecording()'s init, not a compile-time
   constant (logf() isn't guaranteed to fold at compile time), so this is a variable, not a
   #define - matches triggerSampleMultiplier's own "computed once at init" pattern below. */

static float triggerLogitThreshold;

#define TRIGGER_TAIL_SECONDS                      5.0f
#define TRIGGER_PREROLL_SECONDS                   0.75f

/* Whole 64ms buffers - the model's/ISR's own granularity - closest to
   ceil(seconds / (MAXIMUM_SAMPLES_IN_DMA_TRANSFER/STFT_SAMPLE_RATE)). */

#define TRIGGER_TAIL_BUFFERS                      79   /* ~5.056s actual */
#define TRIGGER_PREROLL_BUFFERS                   12   /* ~0.768s actual */

/* Power of two (indices masked with & (N-1)); usable capacity is TRIGGER_QUEUE_LENGTH-1 -
   see triggerQueueFreeSlots() - which must stay comfortably above TRIGGER_PREROLL_BUFFERS
   so a trigger onset's whole pre-roll burst can always be enqueued from empty. Under
   sustained SD backlog this can still run out - see triggerQueueFreeSlots()'s call sites
   in the ISR below for the accepted, documented failure mode (a dropped detection or a
   dropped buffer, never a corrupted/inconsistent event sequence). */

#define TRIGGER_QUEUE_LENGTH                      32

#define TRIGGER_EVENT_NONE                        0
#define TRIGGER_EVENT_START                       1
#define TRIGGER_EVENT_CONTINUE                    2
#define TRIGGER_EVENT_END                         3

#pragma pack(push, 1)

/* One record per buffer queued for the main loop to write - either part of a pre-roll
   burst or a normal in-progress buffer. event tells the consumer whether to open a new
   file before writing this buffer's audio (START), just write it (CONTINUE), or write it
   then close the file (END). */

typedef struct {
    uint32_t bufferIndex;
    int16_t audio[MAXIMUM_SAMPLES_IN_DMA_TRANSFER];
    uint8_t event;
} triggerRecord_t;

#pragma pack(pop)

/* External SRAM layout for this mode (see model_inference.c's ACTIVATION_BUFFER_BASE_OFFSET
   comment for the full picture across all three of this firmware's mutually-exclusive
   modes): pre-roll ring at offset 0 (24576 bytes), trigger queue right after at offset
   24576 (65792 bytes, ending at 90368), the model's own activation buffers past that. */

#define TRIGGER_PREROLL_RING_BASE_OFFSET          0
#define TRIGGER_QUEUE_BASE_OFFSET                 24576

/* Pre-roll ring: written unconditionally every ISR call, regardless of trigger state, so
   the TRIGGER_PREROLL_BUFFERS buffers just before a detection are always available to flush
   into a freshly-opened file - too big for internal RAM (24576 bytes). */

static int16_t (*triggerPrerollRing)[MAXIMUM_SAMPLES_IN_DMA_TRANSFER] =
    (int16_t(*)[MAXIMUM_SAMPLES_IN_DMA_TRANSFER])(AM_EXTERNAL_SRAM_START_ADDRESS + TRIGGER_PREROLL_RING_BASE_OFFSET);

static volatile uint32_t triggerPrerollWriteIndex;

/* Set once in runTriggeredRecording()'s init (see its own comment on why) - matches
   makeRecording()'s DigitalFilter_setAdditionalGain(sampleMultiplier), just applied
   directly to the samples here instead of folded into a DigitalFilter gain coefficient,
   since this mode never calls DigitalFilter_applyFilter() at all. */

static float triggerSampleMultiplier;

/* Scales and clamps one buffer's raw ADC samples by triggerSampleMultiplier before they
   reach the WAV file - never applied to what feeds computeSpectrogram()/ModelInference_run()
   (see triggerSampleMultiplier's own comment). */

static void copyScaledAudioForWav(int16_t *dst, const int16_t *src) {

    for (uint32_t i = 0; i < MAXIMUM_SAMPLES_IN_DMA_TRANSFER; i += 1) {

        float scaled = (float)src[i] * triggerSampleMultiplier;

        if (scaled > (float)INT16_MAX) scaled = (float)INT16_MAX;

        if (scaled < (float)(-INT16_MAX)) scaled = (float)(-INT16_MAX);

        dst[i] = (int16_t)scaled;

    }

}

/* Producer (ISR) / consumer (main loop) queue - same pattern as STFT_DEBUG_MODE's
   stftRecordQueue, just carrying event-tagged raw audio instead of debug/timing data. */

static triggerRecord_t *triggerQueue = (triggerRecord_t*)(AM_EXTERNAL_SRAM_START_ADDRESS + TRIGGER_QUEUE_BASE_OFFSET);

static volatile uint32_t triggerQueueWriteIndex;

static volatile uint32_t triggerQueueReadIndex;

static volatile uint32_t triggerBufferCounter;

/* Free slots left in triggerQueue, reserving one slot always empty so a genuinely full
   queue (write index having wrapped all the way back around to the read index) can never
   be numerically mistaken for an empty one - the classic power-of-two ring buffer
   ambiguity, avoided here by never letting occupancy reach the full TRIGGER_QUEUE_LENGTH. */

static uint32_t triggerQueueFreeSlots(void) {

    uint32_t usedSlots = (triggerQueueWriteIndex - triggerQueueReadIndex) & (TRIGGER_QUEUE_LENGTH - 1);

    return (TRIGGER_QUEUE_LENGTH - 1) - usedSlots;

}

inline void AudioMoth_handleDirectMemoryAccessInterrupt(bool isPrimaryBuffer, int16_t **nextBuffer) {

    int16_t *source = isPrimaryBuffer ? primaryBuffer : secondaryBuffer;

    computeSpectrogram(source);

    float logit = ModelInference_run(stftSpectrogramDb, NULL);

    /* Pre-roll ring: always updated, regardless of trigger state - after this copy its
       newest entry is always the current buffer. Scaled for the WAV file here (see
       triggerSampleMultiplier's comment) - computeSpectrogram()/ModelInference_run() above
       already ran on the unscaled source, which is what they need. */

    copyScaledAudioForWav(triggerPrerollRing[triggerPrerollWriteIndex], source);

    triggerPrerollWriteIndex = (triggerPrerollWriteIndex + 1) % TRIGGER_PREROLL_BUFFERS;

    static bool recordingActive;

    static uint32_t tailCountdown;

    bool triggered = logit > triggerLogitThreshold;

    if (recordingActive == false) {

        /* Trigger onset: flush the whole pre-roll ring (walking forward from the slot
           about to be overwritten next - the oldest live entry - wrapping around to end on
           the buffer just written above, i.e. oldest to newest) as one all-or-nothing
           burst, so the consumer never sees a CONTINUE/END record with no preceding START -
           either the whole burst is enqueued, or none of it is. */

        if (triggered && triggerQueueFreeSlots() >= TRIGGER_PREROLL_BUFFERS) {

            for (uint32_t i = 0; i < TRIGGER_PREROLL_BUFFERS; i += 1) {

                uint32_t ringIndex = (triggerPrerollWriteIndex + i) % TRIGGER_PREROLL_BUFFERS;

                triggerQueue[triggerQueueWriteIndex].bufferIndex = triggerBufferCounter;

                memcpy(triggerQueue[triggerQueueWriteIndex].audio, triggerPrerollRing[ringIndex], sizeof(triggerQueue[0].audio));

                triggerQueue[triggerQueueWriteIndex].event = (i == 0) ? TRIGGER_EVENT_START : TRIGGER_EVENT_CONTINUE;

                triggerQueueWriteIndex = (triggerQueueWriteIndex + 1) & (TRIGGER_QUEUE_LENGTH - 1);

            }

            recordingActive = true;

            tailCountdown = TRIGGER_TAIL_BUFFERS;

        }

    } else if (triggerQueueFreeSlots() >= 1) {

        triggerQueue[triggerQueueWriteIndex].bufferIndex = triggerBufferCounter;

        copyScaledAudioForWav(triggerQueue[triggerQueueWriteIndex].audio, source);

        bool isLastBuffer;

        if (triggered) {

            tailCountdown = TRIGGER_TAIL_BUFFERS;

            isLastBuffer = false;

        } else {

            tailCountdown -= 1;

            isLastBuffer = tailCountdown == 0;

        }

        triggerQueue[triggerQueueWriteIndex].event = isLastBuffer ? TRIGGER_EVENT_END : TRIGGER_EVENT_CONTINUE;

        triggerQueueWriteIndex = (triggerQueueWriteIndex + 1) & (TRIGGER_QUEUE_LENGTH - 1);

        if (isLastBuffer) recordingActive = false;

    }

    /* If neither branch above enqueued anything (idle with no trigger, or the queue was
       too full), nothing happens this buffer - cheap, no wasted SD bandwidth. */

    triggerBufferCounter += 1;

}

/* WAV file state for the triggered recording currently open, if any - persists across
   AudioMoth_sleep() iterations of runTriggeredRecording()'s main loop below, since a single
   file spans many drain cycles, not just one. */

static bool triggerFileOpen;

static uint32_t triggerSamplesWrittenThisFile;

static char triggerFilename[MAXIMUM_FILE_NAME_LENGTH];

static char triggerFoldername[MAXIMUM_FILE_NAME_LENGTH];

/* Opens a new WAV file named from the current time, writes a placeholder header (corrected
   at close time once the true sample count is known - same open-placeholder-header/write-
   data/seek-back/rewrite-header/close pattern makeRecording() uses further below, just
   without its 32KB-buffer-specific "overwrite part of the first audio buffer with the
   header" optimization - not needed here since every record is already a small standalone
   write). No GUANO metadata chunk (deployment/GPS/battery/temperature) for this mode - easy
   follow-up later via the existing writeGuanoData() if wanted. On any failure, flashes the
   red LED and leaves triggerFileOpen false - a transient SD hiccup loses at most this one
   file, not the whole listening session. */

static void openTriggeredWavFile(void) {

    uint32_t currentTime, currentMilliseconds;

    AudioMoth_getTime(&currentTime, &currentMilliseconds);

    generateFolderAndFilename(triggerFoldername, triggerFilename, currentTime, true);

    bool success = true;

    if (configSettings->enableDailyFolders && AudioMoth_doesDirectoryExist(triggerFoldername) == false) {

        success = AudioMoth_makeDirectory(triggerFoldername);

    }

    if (success) success = AudioMoth_openFile(triggerFilename);

    if (success) {

        setHeaderDetails(&wavHeader, STFT_SAMPLE_RATE, 0, 0);

        success = AudioMoth_writeToFile(&wavHeader, sizeof(wavHeader_t));

        if (success == false) AudioMoth_closeFile();

    }

    if (success == false) {

        AudioMoth_setRedLED(true);

        AudioMoth_delay(100);

        AudioMoth_setRedLED(false);

    }

    triggerFileOpen = success;

    triggerSamplesWrittenThisFile = 0;

    /* Solid (not blinking) green LED for the whole time a file is open, unlike the default
       firmware's brief activity flashes - see closeTriggeredWavFile() and drainTriggerQueue()
       for the other two places this must be kept in sync with triggerFileOpen. */

    AudioMoth_setGreenLED(success);

}

/* Finalizes the currently-open file's header with the true sample count and closes it.
   Best-effort: even if a step fails, triggerFileOpen is unconditionally cleared afterwards
   and AudioMoth_closeFile() is still called, so the state machine can never get stuck
   believing a file is open that isn't. */

static void closeTriggeredWavFile(void) {

    bool success = AudioMoth_seekInFile(0);

    if (success) {

        setHeaderDetails(&wavHeader, STFT_SAMPLE_RATE, triggerSamplesWrittenThisFile, 0);

        success = AudioMoth_writeToFile(&wavHeader, sizeof(wavHeader_t));

    }

    AudioMoth_closeFile();

    if (success == false) {

        AudioMoth_setRedLED(true);

        AudioMoth_delay(100);

        AudioMoth_setRedLED(false);

    }

    triggerFileOpen = false;

    AudioMoth_setGreenLED(false);

}

/* Writes every queued record to the currently-open file (opening/closing it as directed by
   each record's event), until the queue is empty. Shared by the normal wake-up drain below
   and the switch-to-USB bail path, so a full drain always happens the same way in both. */

static void drainTriggerQueue(void) {

    while (triggerQueueReadIndex != triggerQueueWriteIndex) {

        triggerRecord_t *record = &triggerQueue[triggerQueueReadIndex];

        if (record->event == TRIGGER_EVENT_START) openTriggeredWavFile();

        if (triggerFileOpen) {

            if (AudioMoth_writeToFile(record->audio, NUMBER_OF_BYTES_IN_SAMPLE * MAXIMUM_SAMPLES_IN_DMA_TRANSFER)) {

                triggerSamplesWrittenThisFile += MAXIMUM_SAMPLES_IN_DMA_TRANSFER;

                AudioMoth_syncFile();

            } else {

                triggerFileOpen = false;

                AudioMoth_setGreenLED(false);

            }

        }

        if (record->event == TRIGGER_EVENT_END && triggerFileOpen) closeTriggeredWavFile();

        triggerQueueReadIndex = (triggerQueueReadIndex + 1) & (TRIGGER_QUEUE_LENGTH - 1);

    }

}

/* Standalone entry point: continuously listen and record a new WAV file per detection
   event - see this mode's top-of-block comment for the overall behaviour. */

static void runTriggeredRecording(void) {

    /* Precompute the Hann analysis window and initialise the FFT - same as
       runSTFTFeasibilityTest() above, see its own comment on why
       arm_rfft_fast_init_256_f32() is called directly instead of the generic
       arm_rfft_fast_init_f32() dispatcher (costs ~69KB of unused twiddle tables otherwise,
       see EXPERIMENTS.md). */

    for (uint32_t i = 0; i < STFT_FFT_LENGTH; i += 1) {

        stftWindow[i] = 0.5f - 0.5f * cosf(STFT_TWO_PI * (float32_t)i / (float32_t)(STFT_FFT_LENGTH - 1));

    }

#if STFT_FFT_LENGTH != 256
#error "arm_rfft_fast_init_256_f32() below is hard-coded for STFT_FFT_LENGTH==256"
#endif

    arm_rfft_fast_init_256_f32(&stftFFTInstance);

    /* On a genuine cold boot straight into this mode, main()'s normal power-up block (which
       seeds deploymentID) never runs - see initialiseDeploymentID()'s own comment for why
       this doesn't risk calling AudioMoth_isInitialPowerUp() twice in the same run. */

    if (AudioMoth_isInitialPowerUp()) {

        initialiseDeploymentID();

    }

    /* Wait for the SD card to be ready before doing anything else - same retry-with-LED
       pattern as runSTFTFeasibilityTest(), just without a single upfront file to open,
       since files are opened per detection event here instead. */

    bool sdCardReady = false;

    while (sdCardReady == false) {

        sdCardReady = AudioMoth_enableFileSystem(AM_SD_CARD_NORMAL_SPEED);

        if (sdCardReady == false) {

            AudioMoth_setRedLED(true);

            AudioMoth_delay(100);

            AudioMoth_setRedLED(false);

            AudioMoth_delay(900);

        }

    }

    /* Enable the microphone at a raw, undecimated 16kHz (no software decimation filter) -
       clockDivider/acquisitionCycles/oversampleRate are fixed at 4/16/1 regardless of
       configSettings (same as runSTFTFeasibilityTest()), since the model was trained on
       exactly this raw sample rate/ADC timing and changing them would feed it something
       different from what it expects. Gain and gain range, unlike those, don't affect the
       model's input assumptions - only loudness - so THOSE come from configSettings, same
       as makeRecording() below, so a triggered recording sounds like what this device's
       configured gain (set via the AudioMoth Config App) actually produces, not a fixed
       AM_GAIN_MEDIUM regardless of what's configured (the bug this replaced: recordings
       came out noticeably quieter than normal deployments using a higher configured gain). */

    AudioMoth_enableExternalSRAM();

    AudioMoth_ignoreExternalMicrophone(false);

    AM_gainRange_t triggerGainRange = configSettings->enableLowGainRange ? AM_LOW_GAIN_RANGE : AM_NORMAL_GAIN_RANGE;

    AM_externalMicrophone_t triggerExternalMicrophone = AudioMoth_enableMicrophone(triggerGainRange, configSettings->gain, 4, 16, 1);

    AudioMoth_initialiseDirectMemoryAccess(primaryBuffer, secondaryBuffer, MAXIMUM_SAMPLES_IN_DMA_TRANSFER);

    /* makeRecording() below applies this same "sample multiplier" to every sample via
       DigitalFilter_setAdditionalGain() (folded into DigitalFilter_applyFilter()'s own
       filter gain) - with oversampleRate/sampleRateDivider fixed at 1/1 for this mode (see
       the comment above), that's a constant 16.0f: the AudioMoth ADC's raw one-shot reading
       (oversampleRate=1, no hardware oversampling) sits at a fixed fraction of the standard
       int16 full-scale convention that oversampleRate=16 (the usual default) reaches
       without any correction, so leaving this unapplied - as this mode did until now, since
       it never calls DigitalFilter_applyFilter() at all, sampling raw via DMA directly -
       made every triggered recording come out ~16x (~24dB) quieter than a normal one at the
       same configured gain. Deliberately applied ONLY to the copies feeding the WAV file
       (see copyScaledAudioForWav() below) - never to what feeds computeSpectrogram()/
       ModelInference_run(), whose calibration (RMS normalization target, INT8 quantization
       scales) is tied to this same raw, unscaled ADC convention the model was trained and
       validated against (see EXPERIMENTS.md) - scaling that input too would be a real
       train/inference mismatch, not a fix. */

    triggerSampleMultiplier = 16.0f / (float)(1 * 1);

    if (AudioMoth_hasInvertedOutput()) triggerSampleMultiplier = -triggerSampleMultiplier;

    if (triggerExternalMicrophone == AM_EXTERNAL_PRESENT_AND_USED) triggerSampleMultiplier = -triggerSampleMultiplier;

    /* See TRIGGER_PROBABILITY_THRESHOLD's comment above - computed once here rather than a
       #define so the ISR never calls logf() itself. */

    triggerLogitThreshold = logf(TRIGGER_PROBABILITY_THRESHOLD / (1.0f - TRIGGER_PROBABILITY_THRESHOLD));

    triggerBufferCounter = 0;

    triggerPrerollWriteIndex = 0;

    triggerQueueWriteIndex = 0;

    triggerQueueReadIndex = 0;

    triggerFileOpen = false;

    AudioMoth_setGreenLED(false);

    ModelInference_reset();

    resetRMSNormalization();

    AudioMoth_startMicrophoneSamples(STFT_SAMPLE_RATE);

    while (true) {

        /* Re-checked every cycle (~64ms), same as runSTFTFeasibilityTest(), so flipping the
           switch to USB at any time lets the device fall through to normal USB handling and
           be reflashed. Disables the microphone FIRST (stops the ISR from enqueueing
           anything further), THEN fully drains whatever is left in the queue into the
           currently-open file (so a last fraction of a second of already-captured audio
           isn't silently discarded), THEN finalizes and closes that file if one is open. */

        if (AudioMoth_getSwitchPosition() == AM_SWITCH_USB) {

            AudioMoth_disableMicrophone();

            drainTriggerQueue();

            if (triggerFileOpen) closeTriggeredWavFile();

            return;

        }

        drainTriggerQueue();

        AudioMoth_sleep();

    }

}

#endif

/* Neither debug mode active - restores the normal recording firmware (STFT_DEBUG_MODE and
   TRIGGERED_RECORDING_MODE are both 0). This #else belongs to the OUTER
   `#if STFT_DEBUG_MODE || TRIGGERED_RECORDING_MODE` opened at the top of this whole shared
   STFT/model-inference block, not the STFT_DEBUG_MODE/TRIGGERED_RECORDING_MODE #if/#elif
   pair just closed above - the normal ISR must NOT be nested inside code that only compiles
   when one of those two modes is active. */

#else

inline void AudioMoth_handleDirectMemoryAccessInterrupt(bool isPrimaryBuffer, int16_t **nextBuffer) {

    int16_t *source = secondaryBuffer;

    if (isPrimaryBuffer) source = primaryBuffer;

    /* Apply filter to samples */

    bool thresholdExceeded = DigitalFilter_applyFilter(source, buffers[writeBuffer] + writeBufferIndex, configSettings->sampleRateDivider, numberOfRawSamplesInDMATransfer);

    numberOfDMATransfers += 1;

    /* Update the current buffer index and write buffer if wait period is over */

    if (numberOfDMATransfers > numberOfDMATransfersToWait) {

        writeIndicator[writeBuffer] |= thresholdExceeded;

        writeBufferIndex += numberOfRawSamplesInDMATransfer / configSettings->sampleRateDivider;

        if (writeBufferIndex == NUMBER_OF_SAMPLES_IN_BUFFER) {

            writeBufferIndex = 0;

            writeBuffer = (writeBuffer + 1) & (NUMBER_OF_BUFFERS - 1);

            writeIndicator[writeBuffer] = false;

        }

    } else if (numberOfDMATransfers == numberOfDMATransfersToWait) {

        AudioMoth_setRedLED(false);

    } else {

        bool state = numberOfDMATransfers % PREPARATION_PERIOD_LED_DIVIDER == 0;

        if (enableLED) AudioMoth_setRedLED(state);

    }

}

#endif

/* AudioMoth USB message handlers */

inline void AudioMoth_usbFirmwareVersionRequested(uint8_t **firmwareVersionPtr) {

    *firmwareVersionPtr = firmwareVersion;

}

inline void AudioMoth_usbFirmwareDescriptionRequested(uint8_t **firmwareDescriptionPtr) {

    *firmwareDescriptionPtr = firmwareDescription;

}

inline void AudioMoth_usbApplicationPacketRequested(uint32_t messageType, uint8_t *transmitBuffer, uint32_t size) {

    /* Copy the current time to the USB packet */

    uint32_t currentTime;

    AudioMoth_getTime(&currentTime, NULL);

    memcpy(transmitBuffer + 1, &currentTime, UINT32_SIZE_IN_BYTES);

    /* Copy the unique ID to the USB packet */

    memcpy(transmitBuffer + 5, (uint8_t*)AM_UNIQUE_ID_START_ADDRESS, AM_UNIQUE_ID_SIZE_IN_BYTES);

    /* Copy the battery state to the USB packet */

    uint32_t supplyVoltage = AudioMoth_getSupplyVoltage();

    AM_batteryState_t batteryState = AudioMoth_getBatteryState(supplyVoltage);

    memcpy(transmitBuffer + 5 + AM_UNIQUE_ID_SIZE_IN_BYTES, &batteryState, 1);

    /* Copy the firmware version to the USB packet */

    memcpy(transmitBuffer + 6 + AM_UNIQUE_ID_SIZE_IN_BYTES, firmwareVersion, AM_FIRMWARE_VERSION_LENGTH);

    /* Copy the firmware description to the USB packet */

    memcpy(transmitBuffer + 6 + AM_UNIQUE_ID_SIZE_IN_BYTES + AM_FIRMWARE_VERSION_LENGTH, firmwareDescription, AM_FIRMWARE_DESCRIPTION_LENGTH);

}

inline void AudioMoth_usbApplicationPacketReceived(uint32_t messageType, uint8_t* receiveBuffer, uint8_t *transmitBuffer, uint32_t size) {

    /* Make persistent configuration settings data structure */

    memcpy(&tempPersistentConfigSettings->firmwareVersion, &firmwareVersion, AM_FIRMWARE_VERSION_LENGTH);

    memcpy(&tempPersistentConfigSettings->firmwareDescription, &firmwareDescription, AM_FIRMWARE_DESCRIPTION_LENGTH);

    memcpy(&tempPersistentConfigSettings->configSettings, receiveBuffer + 1,  sizeof(configSettings_t));

    /* Implement energy saver mode changes */

    if (isEnergySaverMode(&tempPersistentConfigSettings->configSettings)) {

        tempPersistentConfigSettings->configSettings.sampleRate /= 2;
        tempPersistentConfigSettings->configSettings.clockDivider /= 2;
        tempPersistentConfigSettings->configSettings.sampleRateDivider /= 2;

    }

    /* Copy persistent configuration settings to flash */

    uint32_t numberOfBytes = ROUND_UP_TO_MULTIPLE(sizeof(persistentConfigSettings_t), UINT32_SIZE_IN_BYTES);

    bool success = AudioMoth_writeToFlashUserDataPage((uint8_t*)tempPersistentConfigSettings, numberOfBytes);

    if (success) {

        /* Copy the persistent configuration to the USB packet */

        memcpy(transmitBuffer + 1, configSettings, sizeof(configSettings_t));

        /* Revert energy saver mode changes */

        configSettings_t *tempConfigSettings = (configSettings_t*)(transmitBuffer + 1);

        if (isEnergySaverMode(tempConfigSettings)) {

            tempConfigSettings->sampleRate *= 2;
            tempConfigSettings->clockDivider *= 2;
            tempConfigSettings->sampleRateDivider *= 2;

        }   

        /* Set the time */

        AudioMoth_setTime(configSettings->time, USB_CONFIG_TIME_CORRECTION);

        /* Blink the green LED */

        AudioMoth_blinkDuringUSB(USB_CONFIGURATION_BLINK);

    } else {

        /* Return blank configuration as error indicator */

        memset(transmitBuffer + 1, 0, sizeof(configSettings_t));

    }

}

/* Audio configuration handlers */

inline void AudioConfig_handleAudioConfigurationEvent(AC_audioConfigurationEvent_t event) {

    if (event == AC_EVENT_PULSE) {

        audioConfigPulseCounter = (audioConfigPulseCounter + 1) % AUDIO_CONFIG_PULSE_INTERVAL;

    } else if (event == AC_EVENT_START) {

        audioConfigStateLED = true;

        audioConfigToggleLED = true;

        AudioMoth_getTime(&secondsOfAcousticSignalStart, &millisecondsOfAcousticSignalStart);

    } else if (event == AC_EVENT_BYTE) {

        audioConfigToggleLED = !audioConfigToggleLED;

    } else if (event == AC_EVENT_BIT_ERROR || event == AC_EVENT_CRC_ERROR) {

        audioConfigStateLED = false;

    }

    AudioMoth_setGreenLED((audioConfigStateLED && audioConfigToggleLED) || (!audioConfigStateLED && !audioConfigPulseCounter));

}

inline void AudioConfig_handleAudioConfigurationPacket(uint8_t *receiveBuffer, uint32_t size) {

    uint32_t standardPacketSize = UINT32_SIZE_IN_BYTES + UINT16_SIZE_IN_BYTES;

    bool standardPacket = size == standardPacketSize;

    bool hasLocation = size == (standardPacketSize + ACOUSTIC_LOCATION_SIZE_IN_BYTES) || size == (standardPacketSize + DEPLOYMENT_ID_LENGTH + ACOUSTIC_LOCATION_SIZE_IN_BYTES);

    bool hasDeploymentID = size == (standardPacketSize + DEPLOYMENT_ID_LENGTH) || size == (standardPacketSize + DEPLOYMENT_ID_LENGTH + ACOUSTIC_LOCATION_SIZE_IN_BYTES);

    if (configSettings->requireAcousticLocation && hasLocation == false) {

        audioConfigStateLED = false;

        return;

    }

    if (standardPacket || hasLocation || hasDeploymentID) {

        /* Copy time from the packet */

        uint32_t time;

        memcpy(&time, receiveBuffer, UINT32_SIZE_IN_BYTES);

        /* Calculate the time correction */

        uint32_t secondsOfAcousticSignalEnd;

        uint32_t millisecondsOfAcousticSignalEnd;

        AudioMoth_getTime(&secondsOfAcousticSignalEnd, &millisecondsOfAcousticSignalEnd);

        uint32_t millisecondTimeOffset = (secondsOfAcousticSignalEnd - secondsOfAcousticSignalStart) * MILLISECONDS_IN_SECOND + millisecondsOfAcousticSignalEnd - millisecondsOfAcousticSignalStart + AUDIO_CONFIG_TIME_CORRECTION;

        /* Set the time */

        AudioMoth_setTime(time + millisecondTimeOffset / MILLISECONDS_IN_SECOND, millisecondTimeOffset % MILLISECONDS_IN_SECOND);

        /* Set the timezone */

        *acousticTimezoneMinutes = *(int16_t*)(receiveBuffer + UINT32_SIZE_IN_BYTES);

        setBackupFlag(BACKUP_ACOUSTIC_TIMEZONE_RECEIVED, true);

        /* Set acoustic location */

        if (hasLocation) {

            acousticLocation_t location;

            memcpy(&location, receiveBuffer + standardPacketSize, ACOUSTIC_LOCATION_SIZE_IN_BYTES);

            setBackupFlag(BACKUP_ACOUSTIC_LOCATION_RECEIVED, true);

            *acousticLatitude = location.latitude;
            
            *acousticLongitude = location.longitude * ACOUSTIC_LONGITUDE_MULTIPLIER;

        }

        /* Set deployment ID */

        if (hasDeploymentID) {

            copyToBackupDomain((uint32_t*)deploymentID, receiveBuffer + standardPacketSize + (hasLocation ? ACOUSTIC_LOCATION_SIZE_IN_BYTES : 0), DEPLOYMENT_ID_LENGTH);

        }

        /* Indicate success */

        AudioConfig_cancelAudioConfiguration();

        acousticConfigurationPerformed = true;

    }

    /* Reset receive state */

    audioConfigStateLED = false;

}

/* Clear and encode the compression buffer */

static void clearCompressionBuffer() {

    for (uint32_t i = 0; i < COMPRESSION_BUFFER_SIZE_IN_BYTES / NUMBER_OF_BYTES_IN_SAMPLE; i += 1) {

        compressionBuffer[i] = 0;

    }

}

static void encodeCompressionBuffer(uint32_t numberOfCompressedBuffers) {

    for (uint32_t i = 0; i < UINT32_SIZE_IN_BITS; i += 1) {

        compressionBuffer[i] = numberOfCompressedBuffers & 0x01 ? 1 : -1;

        numberOfCompressedBuffers >>= 1;

    }

    for (uint32_t i = UINT32_SIZE_IN_BITS; i < COMPRESSION_BUFFER_SIZE_IN_BYTES / NUMBER_OF_BYTES_IN_SAMPLE; i += 1) {

        compressionBuffer[i] = 0;

    }

}

/* Generate foldername and filename from time */

static void generateFolderAndFilename(char *foldername, char *filename, uint32_t timestamp, bool triggeredRecording) {

    struct tm time;

    int32_t timezoneHours, timezoneMinutes;

    AudioMoth_timezoneRequested(&timezoneHours, &timezoneMinutes);

    time_t rawTime = timestamp + timezoneHours * SECONDS_IN_HOUR + timezoneMinutes * SECONDS_IN_MINUTE;

    gmtime_r(&rawTime, &time);

    sprintf(foldername, "%04d%02d%02d", YEAR_OFFSET + time.tm_year, MONTH_OFFSET + time.tm_mon, time.tm_mday);

    uint32_t length = 0;
    
    if (configSettings->enableDailyFolders) {

        length += sprintf(filename, "%s/", foldername);

    } 
    
    if (configSettings->enableFilenameWithDeviceID) {

        uint8_t *source = memcmp(deploymentID, defaultDeploymentID, DEPLOYMENT_ID_LENGTH) ? deploymentID : (uint8_t*)AM_UNIQUE_ID_START_ADDRESS;

        length += sprintf(filename + length, SERIAL_NUMBER "_", FORMAT_SERIAL_NUMBER(source));

    }
    
    length += sprintf(filename + length, "%s_%02d%02d%02d", foldername, time.tm_hour, time.tm_min, time.tm_sec);

    char *extension = triggeredRecording ? "T.WAV" : ".WAV";

    strcpy(filename + length, extension);

}

/* Save recording to SD card */

static AM_recordingState_t makeRecording(uint32_t timeOfNextRecording, uint32_t recordDuration, uint32_t timeAtWhichToSwitchOffLED, AM_extendedBatteryState_t extendedBatteryState, int32_t temperature, uint32_t *fileOpenTime, uint32_t *fileOpenMilliseconds) {

    /* Initialise buffers */

    writeBuffer = 0;

    writeBufferIndex = 0;

    buffers[0] = (int16_t*)AM_EXTERNAL_SRAM_START_ADDRESS;

    for (uint32_t i = 1; i < NUMBER_OF_BUFFERS; i += 1) {
        buffers[i] = buffers[i - 1] + NUMBER_OF_SAMPLES_IN_BUFFER;
    }

    /* Calculate effective sample rate */

    uint32_t effectiveSampleRate = configSettings->sampleRate / configSettings->sampleRateDivider;

    /* Set up the digital filter */

    uint32_t blockingFilterFrequency = configSettings->disable48HzDCBlockingFilter ? LOW_DC_BLOCKING_FREQ : DEFAULT_DC_BLOCKING_FREQ;

    if (configSettings->lowerFilterFreq == 0 && configSettings->higherFilterFreq == 0) {

        requestedFilterType = NO_FILTER;

        DigitalFilter_designHighPassFilter(effectiveSampleRate, blockingFilterFrequency);

    } else if (configSettings->lowerFilterFreq == UINT16_MAX) {

        requestedFilterType = LOW_PASS_FILTER;

        DigitalFilter_designBandPassFilter(effectiveSampleRate, blockingFilterFrequency, FILTER_FREQ_MULTIPLIER * configSettings->higherFilterFreq);

    } else if (configSettings->higherFilterFreq == UINT16_MAX) {

        requestedFilterType = HIGH_PASS_FILTER;

        DigitalFilter_designHighPassFilter(effectiveSampleRate, MAX(blockingFilterFrequency, FILTER_FREQ_MULTIPLIER * configSettings->lowerFilterFreq));

    } else {

        requestedFilterType = BAND_PASS_FILTER;

        DigitalFilter_designBandPassFilter(effectiveSampleRate, MAX(blockingFilterFrequency, FILTER_FREQ_MULTIPLIER * configSettings->lowerFilterFreq), FILTER_FREQ_MULTIPLIER * configSettings->higherFilterFreq);

    }

    /* Calculate the number of samples in each DMA transfer (while ensuring that number of samples written to the SRAM buffer on each DMA transfer is a power of two so each SRAM buffer is filled after an integer number of DMA transfers) */

    numberOfRawSamplesInDMATransfer = MAXIMUM_SAMPLES_IN_DMA_TRANSFER / configSettings->sampleRateDivider;

    while (numberOfRawSamplesInDMATransfer & (numberOfRawSamplesInDMATransfer - 1)) {

        numberOfRawSamplesInDMATransfer = numberOfRawSamplesInDMATransfer & (numberOfRawSamplesInDMATransfer - 1);

    }

    numberOfRawSamplesInDMATransfer *= configSettings->sampleRateDivider;

    /* Calculate the minimum amplitude threshold duration */

    uint32_t minimumNumberOfTriggeredBuffersToWrite = ROUNDED_UP_DIV(configSettings->minimumTriggerDuration * effectiveSampleRate, NUMBER_OF_SAMPLES_IN_BUFFER);

    /* Calculate LED flash timescale when no triggered buffers are detected */

    uint32_t triggeredRecordingFlashInterval = MAX(1, ROUNDED_DIV(effectiveSampleRate, NUMBER_OF_SAMPLES_IN_BUFFER));

    /* Initialise termination conditions */

    microphoneChanged = false;

    bool supplyVoltageLow = false;

    /* Enable the external SRAM and microphone and initialise direct memory access */

    AudioMoth_enableExternalSRAM();

    AudioMoth_ignoreExternalMicrophone(false);

    AM_gainRange_t gainRange = configSettings->enableLowGainRange ? AM_LOW_GAIN_RANGE : AM_NORMAL_GAIN_RANGE;

    AM_externalMicrophone_t externalMicrophone = AudioMoth_enableMicrophone(gainRange, configSettings->gain, configSettings->clockDivider, configSettings->acquisitionCycles, configSettings->oversampleRate);

    AudioMoth_initialiseDirectMemoryAccess(primaryBuffer, secondaryBuffer, numberOfRawSamplesInDMATransfer);

    /* Calculate the sample multiplier */

    float sampleMultiplier = 16.0f / (float)(configSettings->oversampleRate * configSettings->sampleRateDivider);

    if (AudioMoth_hasInvertedOutput()) sampleMultiplier = -sampleMultiplier;

    if (externalMicrophone == AM_EXTERNAL_PRESENT_AND_USED) sampleMultiplier = -sampleMultiplier;

    DigitalFilter_setAdditionalGain(sampleMultiplier);

    /* Determine if amplitude threshold is enabled */

    bool frequencyTriggerEnabled = configSettings->enableFrequencyTrigger;

    bool amplitudeThresholdEnabled = frequencyTriggerEnabled ? false : configSettings->amplitudeThreshold > 0 || configSettings->enableAmplitudeThresholdDecibelScale || configSettings->enableAmplitudeThresholdPercentageScale;

    /* Configure the digital filter for the appropriate trigger */
    
    if (frequencyTriggerEnabled) {

        uint32_t frequency = MIN(effectiveSampleRate / 2, FILTER_FREQ_MULTIPLIER * configSettings->frequencyTriggerCentreFrequency);

        uint32_t windowLength = MIN(FREQUENCY_TRIGGER_WINDOW_MAXIMUM, MAX(FREQUENCY_TRIGGER_WINDOW_MINIMUM, 1 << configSettings->frequencyTriggerWindowLengthShift));

        float percentageThreshold = (float)configSettings->frequencyTriggerThresholdPercentageMantissa * powf(10.0f, (float)configSettings->frequencyTriggerThresholdPercentageExponent); 

        DigitalFilter_setFrequencyTrigger(windowLength, effectiveSampleRate, frequency, percentageThreshold);

    }

    if (amplitudeThresholdEnabled) {
        
        DigitalFilter_setAmplitudeThreshold(configSettings->amplitudeThreshold);

    }

    /* Set the initial header comment details */

    AM_recordingState_t recordingState = SDCARD_WRITE_ERROR;

    setHeaderDetails(&wavHeader, effectiveSampleRate, 0, 0);

    setHeaderComment(&wavHeader, configSettings, timeOfNextRecording, (uint8_t*)AM_UNIQUE_ID_START_ADDRESS, deploymentID, defaultDeploymentID, extendedBatteryState, temperature, externalMicrophone == AM_EXTERNAL_PRESENT_AND_USED, recordingState, requestedFilterType);

    /* Show LED for SD card activity */

    if (enableLED) AudioMoth_setRedLED(true);

    /* Open a file with the current local time as the name */

    static char filename[MAXIMUM_FILE_NAME_LENGTH];

    static char foldername[MAXIMUM_FILE_NAME_LENGTH];

    generateFolderAndFilename(foldername, filename, timeOfNextRecording, frequencyTriggerEnabled || amplitudeThresholdEnabled);

    if (configSettings->enableDailyFolders) {

        bool directoryExists = AudioMoth_doesDirectoryExist(foldername);

        if (directoryExists == false) FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_makeDirectory(foldername));

    }

    FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_openFile(filename));

    /* Write the header */

    FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_writeToFile(&wavHeader, sizeof(wavHeader_t)));    

    FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_syncFile());

    FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_seekInFile(0));

    AudioMoth_setRedLED(false);

    /* Measure the time difference from the start time */

    uint32_t fileOpenMicroseconds;

    AudioMoth_getTimeMicroseconds(fileOpenTime, &fileOpenMicroseconds);

    *fileOpenMilliseconds = ROUNDED_DIV(fileOpenMicroseconds, MICROSECONDS_IN_MILLISECOND);

    /* Calculate time correction for sample rate due to file header */

    uint32_t numberOfSamplesInHeader = sizeof(wavHeader_t) / NUMBER_OF_BYTES_IN_SAMPLE;

    uint32_t sampleRateTimeOffset = ROUNDED_DIV(numberOfSamplesInHeader * MICROSECONDS_IN_SECOND, effectiveSampleRate);

    /* Calculate time until the recording should start */

    int32_t microsecondsUntilRecordingShouldStart = (int64_t)timeOfNextRecording * MICROSECONDS_IN_SECOND - (int64_t)*fileOpenTime * MICROSECONDS_IN_SECOND - (int64_t)fileOpenMicroseconds - (int64_t)sampleRateTimeOffset;

    /* Calculate the actual recording start time if the intended start has been missed */

    uint32_t timeOffset = microsecondsUntilRecordingShouldStart < 0 ? 1 - microsecondsUntilRecordingShouldStart / MICROSECONDS_IN_SECOND : 0;

    recordDuration = timeOffset >= recordDuration ? 0 : recordDuration - timeOffset;

    microsecondsUntilRecordingShouldStart += (int64_t)timeOffset * MICROSECONDS_IN_SECOND;

    uint32_t actualRecordingStartTime = timeOfNextRecording + timeOffset;

    /* Calculate the period to wait before starting the DMA transfers */

    uint64_t numberOfRawSamplesToWait = ROUNDED_DIV((uint64_t)microsecondsUntilRecordingShouldStart * (uint64_t)configSettings->sampleRate, MICROSECONDS_IN_SECOND);
    
    numberOfDMATransfersToWait = (uint32_t)(numberOfRawSamplesToWait / (uint64_t)numberOfRawSamplesInDMATransfer);

    uint32_t remainingNumberOfRawSamples = (uint32_t)(numberOfRawSamplesToWait % (uint64_t)numberOfRawSamplesInDMATransfer);

    uint32_t remainingMicrosecondsToWait = ROUNDED_DIV(remainingNumberOfRawSamples * MICROSECONDS_IN_SECOND, configSettings->sampleRate);

    /* Calculate updated recording parameters */

    uint32_t maximumNumberOfSeconds = MAXIMUM_WAV_DATA_SIZE / NUMBER_OF_BYTES_IN_SAMPLE / effectiveSampleRate;

    bool fileSizeLimited = (recordDuration > maximumNumberOfSeconds);

    uint32_t numberOfSamples = effectiveSampleRate * (fileSizeLimited ? maximumNumberOfSeconds : recordDuration);

    /* Calculate the number of samples at which LED should be turned off */

    uint32_t numberOfSamplesToSwitchOffLED = UINT32_MAX;

    if (enableLED && timeAtWhichToSwitchOffLED < UINT32_MAX) {

        if (timeAtWhichToSwitchOffLED < actualRecordingStartTime) {

            enableLED = false;

        } else if (timeAtWhichToSwitchOffLED < actualRecordingStartTime + recordDuration) {

            numberOfSamplesToSwitchOffLED = numberOfSamplesInHeader + effectiveSampleRate * (timeAtWhichToSwitchOffLED - actualRecordingStartTime);

        }

    }

    /* Initialise main loop variables */

    uint32_t readBuffer = 0;

    uint32_t samplesWritten = 0;

    uint32_t buffersProcessed = 0;

    uint32_t numberOfCompressedBuffers = 0;

    uint32_t totalNumberOfCompressedSamples = 0;

    uint32_t numberOfTriggeredBuffersWritten = 0;

    uint32_t numberOfBuffersProcessedSinceLastWrite = 0;

    bool triggerHasOccurred = false;

    /* Start processing DMA transfers */

    numberOfDMATransfers = 0;

    if (remainingMicrosecondsToWait > DELAY_CORRECTION_MICROSECONDS) {

        remainingMicrosecondsToWait -= DELAY_CORRECTION_MICROSECONDS;

        AudioMoth_delayMicroseconds(remainingMicrosecondsToWait);

    }

    AudioMoth_startMicrophoneSamples(configSettings->sampleRate);

    /* Main recording loop */

    while (samplesWritten < numberOfSamples + numberOfSamplesInHeader && !microphoneChanged && !switchPositionChanged && !magneticSwitch && !supplyVoltageLow) {

        while (readBuffer != writeBuffer && samplesWritten < numberOfSamples + numberOfSamplesInHeader && !microphoneChanged && !switchPositionChanged && !magneticSwitch && !supplyVoltageLow) {

            /* Determine the appropriate number of bytes to the SD card */

            uint32_t numberOfSamplesToWrite = MIN(numberOfSamples + numberOfSamplesInHeader - samplesWritten, NUMBER_OF_SAMPLES_IN_BUFFER);

            /* Check if this buffer should actually be written to the SD card */

            bool writeIndicated = (amplitudeThresholdEnabled == false && frequencyTriggerEnabled == false) || writeIndicator[readBuffer];

            if (frequencyTriggerEnabled && configSettings->sampleRateDivider > 1) writeIndicated = DigitalFilter_applyFrequencyTrigger(buffers[readBuffer], NUMBER_OF_SAMPLES_IN_BUFFER);

            /* Ensure the minimum number of buffers will be written */

            triggerHasOccurred |= writeIndicated;

            numberOfTriggeredBuffersWritten = writeIndicated ? 0 : numberOfTriggeredBuffersWritten + 1;

            bool shouldWriteThisSector = writeIndicated || (triggerHasOccurred && numberOfTriggeredBuffersWritten < minimumNumberOfTriggeredBuffersToWrite);

            /* Compress the buffer or write the buffer to SD card */

            if (shouldWriteThisSector == false && buffersProcessed > 0 && numberOfSamplesToWrite == NUMBER_OF_SAMPLES_IN_BUFFER) {

                /* Increment the number of compressed buffers */

                numberOfCompressedBuffers += NUMBER_OF_BYTES_IN_SAMPLE * NUMBER_OF_SAMPLES_IN_BUFFER / COMPRESSION_BUFFER_SIZE_IN_BYTES;

                /* Show LED at appropriate interval if no SD card writing has occurred */

                numberOfBuffersProcessedSinceLastWrite += 1;

                if (numberOfBuffersProcessedSinceLastWrite % triggeredRecordingFlashInterval == 0) {

                    if (enableLED) AudioMoth_setRedLED(true);

                    AudioMoth_delay(TRIGGERED_RECORDING_FLASH_DURATION);

                    AudioMoth_setRedLED(false);

                }

            } else {

                /* Light LED during SD card write if appropriate */

                if (enableLED) AudioMoth_setRedLED(true);

                /* Encode and write compression buffer */

                if (numberOfCompressedBuffers > 0) {

                    encodeCompressionBuffer(numberOfCompressedBuffers);

                    totalNumberOfCompressedSamples += (numberOfCompressedBuffers - 1) * COMPRESSION_BUFFER_SIZE_IN_BYTES / NUMBER_OF_BYTES_IN_SAMPLE;

                    FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_writeToFile(compressionBuffer, COMPRESSION_BUFFER_SIZE_IN_BYTES));

                    numberOfCompressedBuffers = 0;

                }

                /* Either write the buffer or write a blank buffer */

                if (shouldWriteThisSector) {

                    if (buffersProcessed == 0) memcpy(buffers[readBuffer], &wavHeader, sizeof(wavHeader_t));

                    FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_writeToFile(buffers[readBuffer], NUMBER_OF_BYTES_IN_SAMPLE * numberOfSamplesToWrite));

                } else {

                    clearCompressionBuffer();

                    uint32_t blankBuffersWritten = 0;

                    uint32_t numberOfBlankSamplesToWrite = numberOfSamplesToWrite;

                    while (numberOfBlankSamplesToWrite > 0) {

                        uint32_t numberOfSamples = MIN(numberOfBlankSamplesToWrite, COMPRESSION_BUFFER_SIZE_IN_BYTES / NUMBER_OF_BYTES_IN_SAMPLE);

                        bool writeHeader = buffersProcessed == 0 && blankBuffersWritten == 0 && numberOfSamples >= sizeof(wavHeader_t) / NUMBER_OF_BYTES_IN_SAMPLE;

                        if (writeHeader) memcpy(compressionBuffer, &wavHeader, sizeof(wavHeader_t));

                        FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_writeToFile(compressionBuffer, NUMBER_OF_BYTES_IN_SAMPLE * numberOfSamples));
                            
                        if (writeHeader) memset(compressionBuffer, 0, sizeof(wavHeader_t));

                        numberOfBlankSamplesToWrite -= numberOfSamples;

                        blankBuffersWritten += 1;

                    }

                }

                /* Clear LED */

                AudioMoth_setRedLED(false);

                /* Reset the buffer counter */

                numberOfBuffersProcessedSinceLastWrite = 0;

            }

            /* Increment buffer counters */

            readBuffer = (readBuffer + 1) & (NUMBER_OF_BUFFERS - 1);

            samplesWritten += numberOfSamplesToWrite;

            buffersProcessed += 1;

        }

        /* Check the voltage level */

        if (configSettings->enableLowVoltageCutoff && AudioMoth_isSupplyAboveThreshold() == false) {

            supplyVoltageLow = true;

        }

        /* Turn LED off if appropriate time has passed */

        if (samplesWritten > numberOfSamplesToSwitchOffLED) enableLED = false;

        /* Sleep until next DMA transfer is complete */

        AudioMoth_sleep();

    }

    /* Write the compression buffer files at the end */

    if (samplesWritten < numberOfSamples + numberOfSamplesInHeader && numberOfCompressedBuffers > 0) {

        /* Light LED during SD card write if appropriate */

        if (enableLED) AudioMoth_setRedLED(true);

        /* Encode and write compression buffer */

        encodeCompressionBuffer(numberOfCompressedBuffers);

        totalNumberOfCompressedSamples += (numberOfCompressedBuffers - 1) * COMPRESSION_BUFFER_SIZE_IN_BYTES / NUMBER_OF_BYTES_IN_SAMPLE;

        FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_writeToFile(compressionBuffer, COMPRESSION_BUFFER_SIZE_IN_BYTES));

        /* Clear LED */

        AudioMoth_setRedLED(false);

    }

    /* Determine recording state */

    recordingState = microphoneChanged ? MICROPHONE_CHANGED :
                     switchPositionChanged ? SWITCH_CHANGED :
                     magneticSwitch ? MAGNETIC_SWITCH :
                     supplyVoltageLow ? SUPPLY_VOLTAGE_LOW :
                     fileSizeLimited ? FILE_SIZE_LIMITED :
                     RECORDING_OKAY;

    /* Generate the new file name if necessary */
    
    static char newFilename[MAXIMUM_FILE_NAME_LENGTH];

    if (timeOffset > 0) {

        generateFolderAndFilename(foldername, newFilename, actualRecordingStartTime, frequencyTriggerEnabled || amplitudeThresholdEnabled);

    }

    /* Write the GUANO data */

    bool gpsLocationReceived = getBackupFlag(BACKUP_GPS_LOCATION_RECEIVED);

    bool acousticLocationReceived = getBackupFlag(BACKUP_ACOUSTIC_LOCATION_RECEIVED);

    uint32_t guanoDataSize = writeGuanoData((char*)compressionBuffer, configSettings, actualRecordingStartTime, gpsLocationReceived, gpsLastFixLatitude, gpsLastFixLongitude, acousticLocationReceived, acousticLatitude, acousticLongitude, firmwareDescription, firmwareVersion, (uint8_t*)AM_UNIQUE_ID_START_ADDRESS, deploymentID, defaultDeploymentID, timeOffset > 0 ? newFilename : filename, extendedBatteryState, temperature, requestedFilterType);

    FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_writeToFile(compressionBuffer, guanoDataSize));

    /* Initialise the WAV header */

    samplesWritten = MAX(numberOfSamplesInHeader, samplesWritten);

    setHeaderDetails(&wavHeader, effectiveSampleRate, samplesWritten - numberOfSamplesInHeader - totalNumberOfCompressedSamples, guanoDataSize);

    setHeaderComment(&wavHeader, configSettings, actualRecordingStartTime, (uint8_t*)AM_UNIQUE_ID_START_ADDRESS, deploymentID, defaultDeploymentID, extendedBatteryState, temperature, externalMicrophone == AM_EXTERNAL_PRESENT_AND_USED, recordingState, requestedFilterType);

    /* Write the header */

    if (enableLED) AudioMoth_setRedLED(true);

    FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_seekInFile(0));

    FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_writeToFile(&wavHeader, sizeof(wavHeader_t)));

    /* Close the file */

    FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_closeFile());

    AudioMoth_setRedLED(false);

    /* Rename the file if necessary */

    if (timeOffset > 0) {

        if (enableLED) AudioMoth_setRedLED(true);

        FLASH_LED_IF_ENABLED_AND_RETURN_ON_ERROR(AudioMoth_renameFile(filename, newFilename));

        AudioMoth_setRedLED(false);

    }

    /* Return recording state */

    return recordingState;

}

/* Determine sunrise and sunset and schedule recording */

static void determineSunriseAndSunsetTimesAndScheduleRecording(uint32_t currentTime) {

    uint32_t scheduleTime = currentTime;

    uint32_t currentTimeOfNextRecording = *timeOfNextRecording;

    uint32_t currentStartOfRecordingPeriod = *startOfRecordingPeriod;

    /* Calculate initial sunrise and sunset time if appropriate */

    if (configSettings->enableSunRecording && *timeOfNextSunriseSunsetCalculation == 0) {

        determineSunriseAndSunsetTimes(scheduleTime);

        determineTimeOfNextSunriseSunsetCalculation(scheduleTime, timeOfNextSunriseSunsetCalculation);

        determineSunriseAndSunsetTimes(*timeOfNextSunriseSunsetCalculation - SECONDS_IN_DAY);

        determineTimeOfNextSunriseSunsetCalculation(scheduleTime, timeOfNextSunriseSunsetCalculation);

        scheduleRecording(scheduleTime, timeOfNextRecording, indexOfNextRecording, durationOfNextRecording, startOfRecordingPeriod, NULL);

        if (*timeOfNextSunriseSunsetCalculation < *timeOfNextRecording) *timeOfNextSunriseSunsetCalculation += SECONDS_IN_DAY;
    
    } else {

        scheduleRecording(scheduleTime, timeOfNextRecording, indexOfNextRecording, durationOfNextRecording, startOfRecordingPeriod, NULL);
  
    }

    /* Check if sunrise and sunset should be recalculated */

    if (configSettings->enableSunRecording && *timeOfNextRecording >= *timeOfNextSunriseSunsetCalculation) {
        
        scheduleTime = MAX(scheduleTime, *timeOfNextSunriseSunsetCalculation);

        determineSunriseAndSunsetTimes(scheduleTime);

        determineTimeOfNextSunriseSunsetCalculation(scheduleTime, timeOfNextSunriseSunsetCalculation);

        scheduleRecording(scheduleTime, timeOfNextRecording, indexOfNextRecording, durationOfNextRecording, startOfRecordingPeriod, NULL);

        if (*timeOfNextSunriseSunsetCalculation < *timeOfNextRecording) *timeOfNextSunriseSunsetCalculation += SECONDS_IN_DAY;

    }

    /* Finished if not using GPS */

    if (configSettings->enableTimeSettingFromGPS == false) return;

    /* Update GPS time setting time */
    
    bool shouldSetTimeFromGPS = false;

    uint32_t gpsTimeSettingPeriod = configSettings->gpsTimeSettingPeriod == 0 ? GPS_DEFAULT_TIME_SETTING_PERIOD : configSettings->gpsTimeSettingPeriod * SECONDS_IN_MINUTE;

    if (configSettings->enableTimeSettingBeforeAndAfterRecordings) {

        *timeOfNextGPSTimeSetting = *timeOfNextRecording - gpsTimeSettingPeriod;

        shouldSetTimeFromGPS = currentTimeOfNextRecording != UINT32_MAX && currentTimeOfNextRecording != *timeOfNextRecording;

    } else {

        *timeOfNextGPSTimeSetting = *startOfRecordingPeriod - gpsTimeSettingPeriod;

        shouldSetTimeFromGPS = currentStartOfRecordingPeriod != UINT32_MAX && currentStartOfRecordingPeriod != *startOfRecordingPeriod;

    }

    setBackupFlag(BACKUP_SHOULD_SET_TIME_FROM_GPS, shouldSetTimeFromGPS);

}

/* Determine sunrise and sunset calculation time */

static void determineTimeOfNextSunriseSunsetCalculation(uint32_t currentTime, uint32_t *timeOfNextSunriseSunsetCalculation) {

    /* Check if limited by earliest recording time */

    uint32_t earliestRecordingTime = configSettings->earliestRecordingTime;

    if (earliestRecordingTime > 0) {

        if (getBackupFlag(BACKUP_ACOUSTIC_TIMEZONE_RECEIVED) && configSettings->useTimezoneFromAcousticChime && configSettings->adjustScheduleUsingTimezoneFromAcousticChime) {

            earliestRecordingTime += configSettings->timezoneHours * SECONDS_IN_HOUR + configSettings->timezoneMinutes * SECONDS_IN_MINUTE - *acousticTimezoneMinutes * SECONDS_IN_MINUTE;

        }

        currentTime = MAX(currentTime, earliestRecordingTime);

    }

    /* Determine when the middle of largest gap between recording periods occurs */

    uint32_t startOfGapMinutes, endOfGapMinutes;

    if (*numberOfSunRecordingPeriods == 1) {

        startOfGapMinutes = firstSunRecordingPeriod->endMinutes;

        endOfGapMinutes = firstSunRecordingPeriod->startMinutes;

    } else {

        uint32_t gapFromFirstPeriodToSecondPeriod = secondSunRecordingPeriod->startMinutes - firstSunRecordingPeriod->endMinutes;

        uint32_t gapFromSecondPeriodsToFirstPeriod = secondSunRecordingPeriod->endMinutes < secondSunRecordingPeriod->startMinutes ? firstSunRecordingPeriod->startMinutes - secondSunRecordingPeriod->endMinutes : MINUTES_IN_DAY - secondSunRecordingPeriod->endMinutes + firstSunRecordingPeriod->startMinutes;

        if (gapFromFirstPeriodToSecondPeriod > gapFromSecondPeriodsToFirstPeriod) {

            startOfGapMinutes = firstSunRecordingPeriod->endMinutes;

            endOfGapMinutes = secondSunRecordingPeriod->startMinutes;

        } else {

            startOfGapMinutes = secondSunRecordingPeriod->endMinutes;

            endOfGapMinutes = firstSunRecordingPeriod->startMinutes;

        }

    }

    uint32_t calculationMinutes = endOfGapMinutes + startOfGapMinutes;

    if (endOfGapMinutes < startOfGapMinutes) calculationMinutes += MINUTES_IN_DAY;

    calculationMinutes /= 2;

    calculationMinutes = calculationMinutes % MINUTES_IN_DAY;

    /* Determine the number of seconds of this day */

    struct tm time;

    time_t rawTime = currentTime;

    gmtime_r(&rawTime, &time);

    uint32_t currentSeconds = SECONDS_IN_HOUR * time.tm_hour + SECONDS_IN_MINUTE * time.tm_min + time.tm_sec;

    /* Determine the time of the next sunrise and sunset calculation */

    uint32_t calculationTime = currentTime - currentSeconds + SECONDS_IN_MINUTE * calculationMinutes;

    if (calculationTime <= currentTime) calculationTime += SECONDS_IN_DAY;

    *timeOfNextSunriseSunsetCalculation = calculationTime;

}

/* Determine sunrise and sunset times */

static void determineSunriseAndSunsetTimes(uint32_t currentTime) {

    /* Calculate future sunrise and sunset time if recording is limited by earliest recording time */

    uint32_t earliestRecordingTime = configSettings->earliestRecordingTime;

    if (earliestRecordingTime > 0) {

        if (getBackupFlag(BACKUP_ACOUSTIC_TIMEZONE_RECEIVED) && configSettings->useTimezoneFromAcousticChime && configSettings->adjustScheduleUsingTimezoneFromAcousticChime) {

            earliestRecordingTime += configSettings->timezoneHours * SECONDS_IN_HOUR + configSettings->timezoneMinutes * SECONDS_IN_MINUTE - *acousticTimezoneMinutes * SECONDS_IN_MINUTE;

        }

        currentTime = MAX(currentTime, earliestRecordingTime);

    }

    /* Determine sunrise and sunset times */
    
    SR_trend_t trend;

    SR_solution_t solution;

    uint32_t sunriseMinutes, sunsetMinutes;

    float latitude = getBackupFlag(BACKUP_GPS_LOCATION_RECEIVED) ? (float)*gpsLatitude / (float)GPS_LOCATION_PRECISION : getBackupFlag(BACKUP_ACOUSTIC_LOCATION_RECEIVED) ? (float)*acousticLatitude / (float)ACOUSTIC_LOCATION_PRECISION : (float)configSettings->latitude / (float)CONFIG_LOCATION_PRECISION;

    float longitude = getBackupFlag(BACKUP_GPS_LOCATION_RECEIVED) ? (float)*gpsLongitude / (float)GPS_LOCATION_PRECISION : getBackupFlag(BACKUP_ACOUSTIC_LOCATION_RECEIVED) ? (float)*acousticLongitude / (float)ACOUSTIC_LOCATION_PRECISION : (float)configSettings->longitude / (float)CONFIG_LOCATION_PRECISION;

    Sunrise_calculateFromUnix(configSettings->sunRecordingEvent, currentTime, latitude, longitude, &solution, &trend, &sunriseMinutes, &sunsetMinutes);

    /* Calculate maximum recording duration */

    uint32_t minimumRecordingGap = MAX(MINIMUM_SUN_RECORDING_GAP, SUN_RECORDING_GAP_MULTIPLIER * configSettings->sunRoundingMinutes);

    uint32_t maximumRecordingDuration = MINUTES_IN_DAY - minimumRecordingGap;

    /* Round the sunrise and sunset times */

    uint32_t roundedSunriseMinutes = configSettings->sunRoundingMinutes > 0 ? UNSIGNED_ROUND(sunriseMinutes, configSettings->sunRoundingMinutes) : sunriseMinutes;

    uint32_t roundedSunsetMinutes = configSettings->sunRoundingMinutes > 0 ? UNSIGNED_ROUND(sunsetMinutes, configSettings->sunRoundingMinutes) : sunsetMinutes;

    /* Calculate start and end of potential recording periods */

    uint32_t beforeSunrise = (MINUTES_IN_DAY + roundedSunriseMinutes - configSettings->beforeSunriseMinutes) % MINUTES_IN_DAY;

    uint32_t afterSunrise = (MINUTES_IN_DAY + roundedSunriseMinutes + configSettings->afterSunriseMinutes) % MINUTES_IN_DAY;

    uint32_t beforeSunset = (MINUTES_IN_DAY + roundedSunsetMinutes - configSettings->beforeSunsetMinutes) % MINUTES_IN_DAY;

    uint32_t afterSunset = (MINUTES_IN_DAY + roundedSunsetMinutes + configSettings->afterSunsetMinutes) % MINUTES_IN_DAY;

    /* Determine schedule */

    recordingPeriod_t tempRecordingPeriod;
    
    if (configSettings->sunRecordingMode == SUNRISE_RECORDING) {

        /* Recording from before sunrise to after sunrise */

        *numberOfSunRecordingPeriods = 1;

        tempRecordingPeriod.startMinutes = beforeSunrise;
            
        tempRecordingPeriod.endMinutes = afterSunrise;
            
        copyToBackupDomain((uint32_t*)firstSunRecordingPeriod, (uint8_t*)&tempRecordingPeriod, sizeof(recordingPeriod_t));

    } else if (configSettings->sunRecordingMode == SUNSET_RECORDING) {

        /* Recording from before sunset to after sunset */

        *numberOfSunRecordingPeriods = 1;

        tempRecordingPeriod.startMinutes = beforeSunset;
            
        tempRecordingPeriod.endMinutes = afterSunset;

        copyToBackupDomain((uint32_t*)firstSunRecordingPeriod, (uint8_t*)&tempRecordingPeriod, sizeof(recordingPeriod_t));

    } else if (configSettings->sunRecordingMode == SUNRISE_AND_SUNSET_RECORDING) {

        /* Order the recording periods */

        uint32_t firstPeriodStartMinutes = beforeSunrise < beforeSunset ? beforeSunrise : beforeSunset;
        
        uint32_t firstPeriodEndMinutes = beforeSunrise < beforeSunset ? afterSunrise : afterSunset;
       
        uint32_t secondPeriodStartMinutes = beforeSunrise < beforeSunset ? beforeSunset : beforeSunrise;
        
        uint32_t secondPeriodEndMinutes  = beforeSunrise < beforeSunset ? afterSunset : afterSunrise;

        /* Determine whether the recording periods wrap */

        bool firstPeriodWraps = firstPeriodEndMinutes <= firstPeriodStartMinutes;

        bool secondPeriodWraps = secondPeriodEndMinutes <= secondPeriodStartMinutes;

        /* Combine recording periods together if they overlap */

        if (firstPeriodWraps) {

            *numberOfSunRecordingPeriods = 1;

            tempRecordingPeriod.startMinutes = firstPeriodStartMinutes;

            tempRecordingPeriod.endMinutes = secondPeriodWraps ? MIN(firstPeriodStartMinutes, MAX(firstPeriodEndMinutes, secondPeriodEndMinutes)) : firstPeriodEndMinutes;

            copyToBackupDomain((uint32_t*)firstSunRecordingPeriod, (uint8_t*)&tempRecordingPeriod, sizeof(recordingPeriod_t));

        } else if (secondPeriodStartMinutes <= firstPeriodEndMinutes) {

            *numberOfSunRecordingPeriods = 1;

            tempRecordingPeriod.startMinutes = firstPeriodStartMinutes;

            tempRecordingPeriod.endMinutes = secondPeriodWraps ? MIN(firstPeriodStartMinutes, secondPeriodEndMinutes) : MAX(firstPeriodEndMinutes, secondPeriodEndMinutes);

            copyToBackupDomain((uint32_t*)firstSunRecordingPeriod, (uint8_t*)&tempRecordingPeriod, sizeof(recordingPeriod_t));

        } else if (secondPeriodWraps && secondPeriodEndMinutes >= firstPeriodStartMinutes) {

            *numberOfSunRecordingPeriods = 1;

            tempRecordingPeriod.startMinutes = secondPeriodStartMinutes;

            tempRecordingPeriod.endMinutes = MAX(firstPeriodEndMinutes, secondPeriodEndMinutes);

            copyToBackupDomain((uint32_t*)firstSunRecordingPeriod, (uint8_t*)&tempRecordingPeriod, sizeof(recordingPeriod_t));

        } else {

            *numberOfSunRecordingPeriods = 2;

            tempRecordingPeriod.startMinutes = firstPeriodStartMinutes;

            tempRecordingPeriod.endMinutes = firstPeriodEndMinutes;

            copyToBackupDomain((uint32_t*)firstSunRecordingPeriod, (uint8_t*)&tempRecordingPeriod, sizeof(recordingPeriod_t));

            tempRecordingPeriod.startMinutes = secondPeriodStartMinutes;

            tempRecordingPeriod.endMinutes = secondPeriodEndMinutes;

            copyToBackupDomain((uint32_t*)secondSunRecordingPeriod, (uint8_t*)&tempRecordingPeriod, sizeof(recordingPeriod_t));

        }

        /* Adjust the size of the minimum gap between recording periods if it is less than the threshold  */

        if (*numberOfSunRecordingPeriods == 1) {

            uint32_t duration = firstSunRecordingPeriod->endMinutes <= firstSunRecordingPeriod->startMinutes ? MINUTES_IN_DAY + firstSunRecordingPeriod->endMinutes - firstSunRecordingPeriod->startMinutes : firstSunRecordingPeriod->endMinutes - firstSunRecordingPeriod->startMinutes;

            if (duration > maximumRecordingDuration) {

                tempRecordingPeriod.startMinutes = firstSunRecordingPeriod->startMinutes;

                tempRecordingPeriod.endMinutes = (firstSunRecordingPeriod->startMinutes + maximumRecordingDuration) % MINUTES_IN_DAY;

                copyToBackupDomain((uint32_t*)firstSunRecordingPeriod, (uint8_t*)&tempRecordingPeriod, sizeof(recordingPeriod_t));

            }

        }
        
        if (*numberOfSunRecordingPeriods == 2) {

            uint32_t gapFromFirstPeriodToSecondPeriod = secondSunRecordingPeriod->startMinutes - firstSunRecordingPeriod->endMinutes;

            uint32_t gapFromSecondPeriodsToFirstPeriod = secondSunRecordingPeriod->endMinutes < secondSunRecordingPeriod->startMinutes ? firstSunRecordingPeriod->startMinutes - secondSunRecordingPeriod->endMinutes : MINUTES_IN_DAY + firstSunRecordingPeriod->startMinutes - secondSunRecordingPeriod->endMinutes;

            if (gapFromFirstPeriodToSecondPeriod >= gapFromSecondPeriodsToFirstPeriod && gapFromFirstPeriodToSecondPeriod < minimumRecordingGap) {

                tempRecordingPeriod.startMinutes = firstSunRecordingPeriod->startMinutes;

                tempRecordingPeriod.endMinutes = (MINUTES_IN_DAY + firstSunRecordingPeriod->endMinutes - minimumRecordingGap + gapFromFirstPeriodToSecondPeriod) % MINUTES_IN_DAY;

                copyToBackupDomain((uint32_t*)firstSunRecordingPeriod, (uint8_t*)&tempRecordingPeriod, sizeof(recordingPeriod_t));

            } else if (gapFromSecondPeriodsToFirstPeriod >= gapFromFirstPeriodToSecondPeriod && gapFromSecondPeriodsToFirstPeriod < minimumRecordingGap) {

                tempRecordingPeriod.startMinutes = secondSunRecordingPeriod->startMinutes;

                tempRecordingPeriod.endMinutes = (MINUTES_IN_DAY + secondSunRecordingPeriod->endMinutes - minimumRecordingGap + gapFromSecondPeriodsToFirstPeriod) % MINUTES_IN_DAY;

                copyToBackupDomain((uint32_t*)secondSunRecordingPeriod, (uint8_t*)&tempRecordingPeriod, sizeof(recordingPeriod_t));

            }

        }

    } else if (configSettings->sunRecordingMode == SUNSET_TO_SUNRISE_RECORDING) {

        /* Recording from before sunset to after sunrise */

        *numberOfSunRecordingPeriods = 1;

        tempRecordingPeriod.startMinutes = beforeSunset;

        uint32_t timeFromSunsetToSunrise;

        if (roundedSunriseMinutes == roundedSunsetMinutes) {

            timeFromSunsetToSunrise = trend == SR_DAY_SHORTER_THAN_NIGHT ? MINUTES_IN_DAY : 0;

        } else {

            timeFromSunsetToSunrise = roundedSunriseMinutes < roundedSunsetMinutes ? MINUTES_IN_DAY + roundedSunriseMinutes - roundedSunsetMinutes : roundedSunriseMinutes - roundedSunsetMinutes;
        
        }
        
        uint32_t duration = timeFromSunsetToSunrise + configSettings->beforeSunsetMinutes + configSettings->afterSunriseMinutes;

        if (duration == 0) duration = 1;

        if (duration > maximumRecordingDuration) duration = maximumRecordingDuration;

        tempRecordingPeriod.endMinutes = (beforeSunset + duration) % MINUTES_IN_DAY;

        copyToBackupDomain((uint32_t*)firstSunRecordingPeriod, (uint8_t*)&tempRecordingPeriod, sizeof(recordingPeriod_t));

    } else if (configSettings->sunRecordingMode == SUNRISE_TO_SUNSET_RECORDING) {

        /* Recording from before sunrise to after sunset */

        *numberOfSunRecordingPeriods = 1;

        tempRecordingPeriod.startMinutes = beforeSunrise;

        uint32_t timeFromSunriseToSunset;

        if (roundedSunriseMinutes == roundedSunsetMinutes) {

            timeFromSunriseToSunset = trend == SR_DAY_LONGER_THAN_NIGHT ? MINUTES_IN_DAY : 0;

        } else {

            timeFromSunriseToSunset = roundedSunsetMinutes < roundedSunriseMinutes ? MINUTES_IN_DAY + roundedSunsetMinutes - roundedSunriseMinutes : roundedSunsetMinutes - roundedSunriseMinutes;

        }

        uint32_t duration = timeFromSunriseToSunset + configSettings->beforeSunriseMinutes + configSettings->afterSunsetMinutes;

        if (duration == 0) duration = 1;

        if (duration > maximumRecordingDuration) duration = maximumRecordingDuration;

        tempRecordingPeriod.endMinutes = (beforeSunrise + duration) % MINUTES_IN_DAY;
        
        copyToBackupDomain((uint32_t*)firstSunRecordingPeriod, (uint8_t*)&tempRecordingPeriod, sizeof(recordingPeriod_t));

    }

}

/* Schedule recordings */

static void adjustRecordingDuration(uint32_t *duration, uint32_t recordDuration, uint32_t sleepDuration) {

    uint32_t durationOfCycle = recordDuration + sleepDuration;

    uint32_t numberOfCycles = *duration / durationOfCycle;

    uint32_t partialCycle = *duration % durationOfCycle;

    if (partialCycle == 0) {

        *duration = *duration > sleepDuration ? *duration - sleepDuration : 0;

    } else {

        *duration = MIN(*duration, numberOfCycles * durationOfCycle + recordDuration);

    }

}

static void calculateStartAndDuration(uint32_t currentTime, uint32_t currentSeconds, recordingPeriod_t *period, int32_t startMinutesOffset, uint32_t *startTime, uint32_t *duration) {

    uint32_t startMinutes = (2 * MINUTES_IN_DAY + period->startMinutes + startMinutesOffset) % MINUTES_IN_DAY;

    *startTime = currentTime - currentSeconds + SECONDS_IN_MINUTE * startMinutes;

    *duration = period->endMinutes <= period->startMinutes ? MINUTES_IN_DAY + period->endMinutes - period->startMinutes : period->endMinutes - period->startMinutes;
    
    *duration *= SECONDS_IN_MINUTE;

}

static void scheduleRecording(uint32_t currentTime, uint32_t *timeOfNextRecording, uint32_t *indexOfNextRecording, uint32_t *durationOfNextRecording, uint32_t *startOfRecordingPeriod, uint32_t *endOfRecordingPeriod) {

    /* Enforce minumum schedule date */
    
    currentTime = MAX(currentTime, START_OF_CENTURY);

    /* Check if recording should be limited by earliest recording time */

    uint32_t earliestRecordingTime = configSettings->earliestRecordingTime;

    if (earliestRecordingTime > 0) {

        if (getBackupFlag(BACKUP_ACOUSTIC_TIMEZONE_RECEIVED) && configSettings->useTimezoneFromAcousticChime && configSettings->adjustScheduleUsingTimezoneFromAcousticChime) {

            earliestRecordingTime += configSettings->timezoneHours * SECONDS_IN_HOUR + configSettings->timezoneMinutes * SECONDS_IN_MINUTE - *acousticTimezoneMinutes * SECONDS_IN_MINUTE;

        }

        currentTime = MAX(currentTime, earliestRecordingTime);

    }

    /* Select appropriate recording periods */

    uint32_t activeRecordingPeriods = configSettings->enableSunRecording ? *numberOfSunRecordingPeriods : MIN(configSettings->activeRecordingPeriods, MAX_RECORDING_PERIODS);

    recordingPeriod_t *recordingPeriods = configSettings->enableSunRecording ? firstSunRecordingPeriod : configSettings->recordingPeriods;

    /* No suitable recording periods */

    if (activeRecordingPeriods == 0) {

        *timeOfNextRecording = UINT32_MAX;

        *indexOfNextRecording = 0;

        if (startOfRecordingPeriod) *startOfRecordingPeriod = UINT32_MAX;

        if (endOfRecordingPeriod) *endOfRecordingPeriod = UINT32_MAX;

        *durationOfNextRecording = 0;

        return;

    }

    /* Calculate the number of seconds of this day */

    struct tm time;

    time_t rawTime = currentTime;

    gmtime_r(&rawTime, &time);

    uint32_t currentSeconds = SECONDS_IN_HOUR * time.tm_hour + SECONDS_IN_MINUTE * time.tm_min + time.tm_sec;

    /* Calculate the start time offset for the appropriate time zone */

    uint32_t minimumIndex = 0;

    int32_t startMinutesOffset = 0;

    if (configSettings->enableSunRecording == false) {

        if (getBackupFlag(BACKUP_ACOUSTIC_TIMEZONE_RECEIVED) && configSettings->useTimezoneFromAcousticChime && configSettings->adjustScheduleUsingTimezoneFromAcousticChime) {

            startMinutesOffset = configSettings->timezoneHours * MINUTES_IN_HOUR + configSettings->timezoneMinutes - *acousticTimezoneMinutes;

        }

        /* Find the first recording period */

        uint32_t minimumStartMinutes = UINT32_MAX;

        for (uint32_t i = 0; i < activeRecordingPeriods; i += 1) {

            recordingPeriod_t *currentPeriod = recordingPeriods + i;

            uint32_t startMinutes = (2 * MINUTES_IN_DAY + currentPeriod->startMinutes + startMinutesOffset) % MINUTES_IN_DAY;

            if (startMinutes < minimumStartMinutes) {

                minimumStartMinutes = startMinutes;

                minimumIndex = i;

            }

        }

    }

    /* Check the last active period on the previous day */

    uint32_t startTime, duration;

    uint32_t index = (minimumIndex + activeRecordingPeriods - 1) % activeRecordingPeriods;

    recordingPeriod_t *lastPeriod = recordingPeriods + index;
    
    calculateStartAndDuration(currentTime - SECONDS_IN_DAY, currentSeconds, lastPeriod, startMinutesOffset, &startTime, &duration);

    if (configSettings->disableSleepRecordCycle == false) {
    
        adjustRecordingDuration(&duration, configSettings->recordDuration, configSettings->sleepDuration);

    }

    if (currentTime < startTime + duration && duration > 0) goto done;

    /* Check each active recording period on the same day */

    for (uint32_t i = 0; i < activeRecordingPeriods; i += 1) {

        index = (minimumIndex + i) % activeRecordingPeriods;

        recordingPeriod_t *currentPeriod = recordingPeriods + index;

        calculateStartAndDuration(currentTime, currentSeconds, currentPeriod, startMinutesOffset, &startTime, &duration);

        if (configSettings->disableSleepRecordCycle == false) {
        
            adjustRecordingDuration(&duration, configSettings->recordDuration, configSettings->sleepDuration);

        }
        
        if (currentTime < startTime + duration && duration > 0) goto done;

    }

    /* Calculate time until first period tomorrow */

    index = minimumIndex;

    recordingPeriod_t *firstPeriod = recordingPeriods + index;

    calculateStartAndDuration(currentTime + SECONDS_IN_DAY, currentSeconds, firstPeriod, startMinutesOffset, &startTime, &duration);

    if (configSettings->disableSleepRecordCycle == false) {
    
        adjustRecordingDuration(&duration, configSettings->recordDuration, configSettings->sleepDuration);

    }

done:

    /* Set the time for start and end of the recording period */

    if (startOfRecordingPeriod) *startOfRecordingPeriod = startTime;

    if (endOfRecordingPeriod) *endOfRecordingPeriod = startTime + duration;

    /* Resolve sleep and record cycle */

    if (configSettings->disableSleepRecordCycle) {

        *timeOfNextRecording = startTime;

        *durationOfNextRecording = duration;

    } else {
        
        if (currentTime <= startTime) {

            /* Recording should start at the start of the recording period */

            *timeOfNextRecording = startTime;

            *durationOfNextRecording = MIN(duration, configSettings->recordDuration);

        } else {

            /* Recording should start immediately or at the start of the next recording cycle */

            uint32_t secondsFromStartOfPeriod = currentTime - startTime;

            uint32_t durationOfCycle = configSettings->recordDuration + configSettings->sleepDuration;

            uint32_t partialCycle = secondsFromStartOfPeriod % durationOfCycle;

            *timeOfNextRecording = currentTime - partialCycle;

            if (partialCycle >= configSettings->recordDuration) {

                /* Wait for next cycle to begin */

                *timeOfNextRecording += durationOfCycle;

            }

            uint32_t remainingDuration = startTime + duration - *timeOfNextRecording;

            *durationOfNextRecording = MIN(configSettings->recordDuration, remainingDuration);

        }

    }

    /* Update start time and duration is recording period has started */

    if (currentTime > *timeOfNextRecording) {

        *durationOfNextRecording -= currentTime - *timeOfNextRecording;

        *timeOfNextRecording = currentTime;
        
    }

    /* Check if recording should be limited by last recording time */

    uint32_t latestRecordingTime = MIDPOINT_OF_CENTURY;

    if (configSettings->latestRecordingTime > 0) {

        latestRecordingTime = configSettings->latestRecordingTime;

        if (getBackupFlag(BACKUP_ACOUSTIC_TIMEZONE_RECEIVED) && configSettings->useTimezoneFromAcousticChime && configSettings->adjustScheduleUsingTimezoneFromAcousticChime) {

            latestRecordingTime += configSettings->timezoneHours * SECONDS_IN_HOUR + configSettings->timezoneMinutes * SECONDS_IN_MINUTE - *acousticTimezoneMinutes * SECONDS_IN_MINUTE;

        }

    }

    if (*timeOfNextRecording >= latestRecordingTime) {

        *timeOfNextRecording = UINT32_MAX;

        if (startOfRecordingPeriod) *startOfRecordingPeriod = UINT32_MAX;

        if (endOfRecordingPeriod) *endOfRecordingPeriod = UINT32_MAX;

        *durationOfNextRecording = 0;

    } else {

        int64_t excessTime = (int64_t)*timeOfNextRecording + (int64_t)*durationOfNextRecording - (int64_t)latestRecordingTime;

        if (excessTime > 0) {
            
            *durationOfNextRecording -= excessTime;

            if (endOfRecordingPeriod) *endOfRecordingPeriod = *timeOfNextRecording + *durationOfNextRecording;

        }

    }

    /* Set the index of the next recording */

    *indexOfNextRecording = index;

}

/* Flash LED according to battery life */

static void flashLedToIndicateBatteryLife(void) {

    uint32_t numberOfFlashes = LOW_BATTERY_LED_FLASHES;

    uint32_t supplyVoltage = AudioMoth_getSupplyVoltage();

    if (configSettings->batteryLevelDisplayType == NIMH_LIPO_BATTERY_VOLTAGE) {

        /* Set number of flashes according to battery voltage */

        AM_extendedBatteryState_t batteryState = AudioMoth_getExtendedBatteryState(supplyVoltage);

        if (batteryState > AM_EXT_BAT_4V3) {

            numberOfFlashes = 1;

        } else if (batteryState > AM_EXT_BAT_3V5) {

            numberOfFlashes = AM_EXT_BAT_4V4 - batteryState;

        }

    } else {

        /* Set number of flashes according to battery state */

        AM_batteryState_t batteryState = AudioMoth_getBatteryState(supplyVoltage);

        if (batteryState > AM_BATTERY_LOW) {

            numberOfFlashes = (batteryState >= AM_BATTERY_4V6) ? 4 : (batteryState >= AM_BATTERY_4V4) ? 3 : (batteryState >= AM_BATTERY_4V0) ? 2 : 1;

        }

    }

    /* Flash LED */

    for (uint32_t i = 0; i < numberOfFlashes; i += 1) {

        FLASH_LED(Red, SHORT_LED_FLASH_DURATION)

        if (numberOfFlashes == LOW_BATTERY_LED_FLASHES) {

            AudioMoth_delay(SHORT_LED_FLASH_DURATION);

        } else {

            AudioMoth_delay(LONG_LED_FLASH_DURATION);

        }

    }

}
