// Smart Bin - MCU side.
//
// Owns the Qwiic bus and everything the user sees. Detects an item held in front
// of the sensor, tells the MPU once via on_trigger(), then waits for the MPU to
// push a disposal category back via set_feedback(). If Linux never answers, the
// bin still responds - it just shows "unknown".

#include "Modulino.h"
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

ModulinoDistance distance;
ModulinoBuzzer buzzer;
Arduino_LED_Matrix matrix;

const uint8_t MATRIX_WIDTH = 13;
const uint8_t MATRIX_HEIGHT = 8;

// Presence uses hysteresis, not one threshold. An item resting near the edge
// of the window sits on a single threshold and flips state every poll however
// well it is filtered, which showed up as the sensor being "jittery": the
// estimate hovered around 30 mm and IDLE/ARMING alternated at 10 Hz. Entering
// requires being well inside the window; leaving requires being clearly out.
const int NEAR_ENTER_MM = 45;                 // no nearer than this to arm
const int NEAR_EXIT_MM = 25;                  // ...and nearer than this to disarm
const int FAR_ENTER_MM = 240;                 // no further than this to arm
const int FAR_EXIT_MM = 265;                  // ...and further than this to disarm

// Readings below this are noise rather than a close target: a stray 1 mm
// sample dragged the estimate down by 20 mm and inverted the velocity.
const int MIN_VALID_MM = 15;
const unsigned long HOLD_TIME_MS = 3000;      // how long it must be held still
const unsigned long POLL_INTERVAL_MS = 100;   // 10 Hz
const unsigned long FEEDBACK_TIMEOUT_MS = 4000;
const unsigned long RESULT_HOLD_MS = 5000;    // minimum time the result stays lit

// Categories must match the keys in disposal_rules.yaml. Colour and tone live
// here, not in the rules file, because the MCU decides how things are displayed.
//
// There is no Modulino Pixels on this build, so the colour signal goes to the
// two onboard RGB LEDs instead. That allows only one bit per channel, which is
// ample for three categories plus unknown.
struct CategoryStyle {
  const char* name;
  bool r, g, b;
  unsigned int tone;
};

const CategoryStyle STYLES[] = {
  {"recycle",   false, false, true,   880},  // blue
  {"compost",   false, true,  false,  660},  // green
  {"trash",     true,  true,  true,   440},  // white
  {"hazardous", true,  false, false, 1320},  // red
  {"ewaste",    true,  false, true,  1100},  // magenta
  {"unknown",   true,  true,  false,  220},  // yellow
};
const size_t STYLE_COUNT = sizeof(STYLES) / sizeof(STYLES[0]);
const size_t UNKNOWN_STYLE = STYLE_COUNT - 1;

// The matrix is monochrome, so it carries shape while the LEDs carry colour.
// Encoding the category twice means the result is still readable when the two
// colours are hard to tell apart. Rows must stay MATRIX_WIDTH characters wide
// and there must be MATRIX_HEIGHT of them; '#' lights a pixel.
const char* const ICONS[][MATRIX_HEIGHT] = {
  {  // recycle - a thick loop with an arrowhead on the right
    "....#####....",
    "..###...###..",
    ".##.......##.",
    "##.........##",
    "##.......####",
    ".##.......##.",
    "..###...###..",
    "....#####....",
  },
  {  // compost - a filled leaf with a stem
    "........#####",
    "......#######",
    "....#########",
    "..##########.",
    ".#########...",
    "#######......",
    "..#..........",
    ".#...........",
  },
  {  // trash - a bin
    "....#####....",
    "..#########..",
    "..#.......#..",
    "..#.#.#.#.#..",
    "..#.#.#.#.#..",
    "..#.#.#.#.#..",
    "..#.......#..",
    "...#######...",
  },
  {  // hazardous - exclamation mark
    ".....###.....",
    ".....###.....",
    ".....###.....",
    ".....###.....",
    ".....###.....",
    ".............",
    ".....###.....",
    ".....###.....",
  },
  {  // ewaste - a filled lightning bolt
    "........####.",
    ".......####..",
    "......####...",
    ".....########",
    "..########...",
    ".....####....",
    "....####.....",
    "...####......",
  },
  {  // unknown - question mark
    "....#####....",
    "...#.....#...",
    ".........#...",
    "......###....",
    "......#......",
    ".............",
    "......#......",
    ".............",
  },
};

enum State { IDLE, ARMING, WAITING, SHOWING };

// Distance filter.
//
// The ToF sensor is noisy and drops readings outright, returning 0 when it has
// no valid target. Thresholding raw samples meant one bad sample looked exactly
// like a withdrawn item and restarted the three-second hold. A small
// constant-velocity Kalman filter tracks distance and its rate of change, so a
// dropped sample coasts on the prediction instead, and the threshold is applied
// to the estimate rather than the raw reading.
//
// dt is one poll tick throughout, which is why the matrix algebra below reduces
// to additions: velocity is in mm per tick, not mm per second.
const float MEAS_VAR = 100.0f;    // sensor noise, about 10 mm sigma
const float PROC_VAR = 25.0f;     // how much true distance can change per tick
const float GATE_SIGMA = 4.0f;    // reject jumps wilder than this as outliers
const int MAX_MISSES = 3;         // ~300 ms unusable before the item counts as gone

// Streams raw and filtered distance to the Linux log while an item is in play,
// so the filter can be tuned against real sensor behaviour rather than guessed
// at. Off costs nothing; on costs one notify per poll.
const bool DEBUG_DISTANCE = true;

float kx = 0.0f;                  // estimated distance, mm
float kv = 0.0f;                  // estimated velocity, mm per tick
float kP[2][2] = {{0, 0}, {0, 0}};
bool kInit = false;
int kMisses = 0;

void kalmanReset() {
  kInit = false;
  kMisses = 0;
  kx = 0.0f;
  kv = 0.0f;
}

void kalmanStart(float z) {
  kx = z;
  kv = 0.0f;
  kP[0][0] = MEAS_VAR; kP[0][1] = 0.0f;
  kP[1][0] = 0.0f;     kP[1][1] = MEAS_VAR;
  kInit = true;
  kMisses = 0;
}

void kalmanPredict() {
  kx += kv;
  // P = F P F' + Q, with F = {{1,1},{0,1}}. Order matters: each line below
  // reads covariance terms the later lines overwrite.
  kP[0][0] += kP[0][1] + kP[1][0] + kP[1][1] + PROC_VAR * 0.25f;
  kP[0][1] += kP[1][1] + PROC_VAR * 0.5f;
  kP[1][0] += kP[1][1] + PROC_VAR * 0.5f;
  kP[1][1] += PROC_VAR;
}

bool kalmanUpdate(float z) {
  float y = z - kx;
  float S = kP[0][0] + MEAS_VAR;

  // Gate on innovation. A genuine fast withdrawal also trips this, but the
  // miss counter resets the filter after a few, so it re-acquires rather than
  // locking onto a stale estimate.
  if (y * y > GATE_SIGMA * GATE_SIGMA * S) {
    return false;
  }

  float K0 = kP[0][0] / S;
  float K1 = kP[1][0] / S;

  kx += K0 * y;
  kv += K1 * y;

  float p00 = kP[0][0];
  float p01 = kP[0][1];
  kP[0][0] -= K0 * p00;
  kP[0][1] -= K0 * p01;
  kP[1][0] -= K1 * p00;
  kP[1][1] -= K1 * p01;
  return true;
}

State state = IDLE;
bool itemPresent = false;
int lastDistanceMm = 0;   // filtered estimate, sent with on_trigger
unsigned long lastPollTime = 0;
unsigned long holdStartTime = 0;
unsigned long triggerSentTime = 0;
unsigned long resultShownTime = 0;

bool mpuReady = false;
bool feedbackPending = false;
String pendingCategory = "unknown";

// Bridge handlers only record what arrived; loop() drives the hardware so that
// neither the Qwiic bus nor the LEDs are touched from two contexts at once.
void set_feedback(String category) {
  pendingCategory = category;
  feedbackPending = true;
}

void mpu_ready() {
  mpuReady = true;
}

// Echoes a line from the Linux side onto the serial monitor. The classifier
// runs on the MPU, so its per-frame output would otherwise only appear in the
// Python console; this puts it beside the MCU's own state changes.
void mcu_log(String line) {
  Monitor.println(line);
}

void setLeds(bool r, bool g, bool b) {
  // The onboard RGB LEDs are active low: LOW lights a channel.
  digitalWrite(LED3_R, r ? LOW : HIGH);
  digitalWrite(LED3_G, g ? LOW : HIGH);
  digitalWrite(LED3_B, b ? LOW : HIGH);
  digitalWrite(LED4_R, r ? LOW : HIGH);
  digitalWrite(LED4_G, g ? LOW : HIGH);
  digitalWrite(LED4_B, b ? LOW : HIGH);
}

void showIcon(size_t style) {
  static uint8_t pixels[MATRIX_HEIGHT * MATRIX_WIDTH];
  size_t n = 0;

  for (uint8_t y = 0; y < MATRIX_HEIGHT; y++) {
    const char* row = ICONS[style][y];
    for (uint8_t x = 0; x < MATRIX_WIDTH; x++) {
      pixels[n++] = (row[x] == '#') ? 1 : 0;
    }
  }

  matrix.loadPixels(pixels, n);
}

void clearDisplay() {
  setLeds(false, false, false);
  matrix.clear();
}

size_t styleIndexFor(const String& name) {
  for (size_t i = 0; i < STYLE_COUNT; i++) {
    if (name == STYLES[i].name) {
      return i;
    }
  }
  return UNKNOWN_STYLE;
}

void showCategory(const String& name) {
  size_t index = styleIndexFor(name);
  const CategoryStyle& style = STYLES[index];

  setLeds(style.r, style.g, style.b);
  showIcon(index);
  buzzer.tone(style.tone, 200);

  Monitor.print(">>> category: ");
  Monitor.println(name);
}

void setup() {
  Bridge.begin();
  Monitor.begin();
  delay(1000);

  pinMode(LED3_R, OUTPUT);
  pinMode(LED3_G, OUTPUT);
  pinMode(LED3_B, OUTPUT);
  pinMode(LED4_R, OUTPUT);
  pinMode(LED4_G, OUTPUT);
  pinMode(LED4_B, OUTPUT);

  matrix.begin();
  clearDisplay();

  Modulino.begin();
  distance.begin();
  buzzer.begin();

  Bridge.provide_safe("set_feedback", set_feedback);
  Bridge.provide_safe("mpu_ready", mpu_ready);
  Bridge.provide_safe("mcu_log", mcu_log);

  Monitor.println("====================================");
  Monitor.println("  SMART BIN MCU READY               ");
  Monitor.println("====================================");
}

void loop() {
  unsigned long now = millis();

  if (feedbackPending) {
    feedbackPending = false;
    if (state == WAITING) {
      showCategory(pendingCategory);
      resultShownTime = now;
      state = SHOWING;
    }
  }

  if (state == WAITING && now - triggerSentTime >= FEEDBACK_TIMEOUT_MS) {
    Monitor.println(">>> no answer from Linux, showing unknown");
    showCategory("unknown");
    resultShownTime = now;
    state = SHOWING;
  }

  if (now - lastPollTime < POLL_INTERVAL_MS) {
    return;
  }
  lastPollTime = now;

  if (distance.available()) {
    int raw_mm = distance.get();

    // 0 means the sensor had no valid target. Any other reading is real, even
    // one far outside the window - it is the threshold's job to decide whether
    // that counts as an item, not the filter's.
    bool usable = (raw_mm >= MIN_VALID_MM && raw_mm < 4000);

    if (!kInit) {
      if (usable) {
        kalmanStart(raw_mm);
      }
    } else {
      kalmanPredict();
      if (usable && kalmanUpdate(raw_mm)) {
        kMisses = 0;
      } else {
        kMisses++;
      }
    }

    if (kInit && kMisses <= MAX_MISSES) {
      if (itemPresent) {
        itemPresent = (kx >= NEAR_EXIT_MM && kx <= FAR_EXIT_MM);
      } else {
        itemPresent = (kx >= NEAR_ENTER_MM && kx <= FAR_ENTER_MM);
      }
      if (itemPresent) {
        lastDistanceMm = (int)(kx + 0.5f);
      }
    } else {
      itemPresent = false;
      if (kMisses > MAX_MISSES) {
        kalmanReset();
      }
    }

    if (DEBUG_DISTANCE && (state != IDLE || itemPresent)) {
      Bridge.notify("host_log",
                    "[mcu] raw=" + String(raw_mm) +
                    " est=" + String(kInit ? (int)(kx + 0.5f) : -1) +
                    " v=" + String(kv, 1) +
                    " miss=" + String(kMisses) +
                    " present=" + String(itemPresent ? 1 : 0) +
                    " state=" + String((int)state));
    }
  }

  // The result stays lit while the item is still held, which is exactly when
  // the user is deciding where to put it. Requiring withdrawal before clearing
  // also stops one item being classified over and over.
  if (state == SHOWING && now - resultShownTime >= RESULT_HOLD_MS && !itemPresent) {
    clearDisplay();
    state = IDLE;
  }

  switch (state) {
    case IDLE:
      if (itemPresent) {
        holdStartTime = now;
        state = ARMING;
        Monitor.println(">>> item detected, holding...");
      }
      break;

    case ARMING:
      if (!itemPresent) {
        state = IDLE;
        Monitor.println("item withdrawn, resetting");
      } else if (now - holdStartTime >= HOLD_TIME_MS) {
        buzzer.tone(1000, 300);
        if (mpuReady) {
          // The distance goes with the trigger: an item at arm's length fills
          // far less of the frame than one held close, and the MPU uses this
          // to crop proportionally.
          Bridge.notify("on_trigger", lastDistanceMm);
          triggerSentTime = now;
          state = WAITING;
          Monitor.print(">>> held 3s at ");
          Monitor.print(lastDistanceMm);
          Monitor.println(" mm, asked Linux to classify");
        } else {
          Monitor.println(">>> held 3s, but Linux is not up yet");
          showCategory("unknown");
          resultShownTime = now;
          state = SHOWING;
        }
      }
      break;

    case WAITING:
    case SHOWING:
      break;
  }
}
