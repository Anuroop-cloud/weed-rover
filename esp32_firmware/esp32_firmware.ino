// ============================================================
// ESP32-P4 + TB6612FNG + 8x8 MATRIX
//
// DEFAULT = 00
//
// 00 = LED OFF / MOTOR OFF
// 01 = LED OFF / MOTOR ON
// 10 = LED ON  / MOTOR OFF
// 11 = LED ON  / MOTOR ON
// ============================================================


// ================= MOTOR =================

#define AIN1 24
#define AIN2 25
#define PWMA 32

#define BIN1 47
#define BIN2 48
#define PWMB 27


// ================= MATRIX =================

const int rowPins[8] = {
  2, 3, 4, 5, 6, 11, 15, 16
};

const int colPins[8] = {
  17, 18, 19, 54, 20, 21, 22, 23
};


// ================= STATE =================

bool ledState = false;
bool motorState = false;

int currentRow = 0;

unsigned long lastScan = 0;


// ================= MOTOR A =================

void motorA(int speed) {

  speed = constrain(speed, -255, 255);

  if (speed > 0) {

    digitalWrite(AIN1, HIGH);
    digitalWrite(AIN2, LOW);
    analogWrite(PWMA, speed);

  } 
  else if (speed < 0) {

    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, HIGH);
    analogWrite(PWMA, -speed);

  } 
  else {

    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, LOW);
    analogWrite(PWMA, 0);
  }
}


// ================= MOTOR B =================

void motorB(int speed) {

  speed = constrain(speed, -255, 255);

  if (speed > 0) {

    digitalWrite(BIN1, HIGH);
    digitalWrite(BIN2, LOW);
    analogWrite(PWMB, speed);

  } 
  else if (speed < 0) {

    digitalWrite(BIN1, LOW);
    digitalWrite(BIN2, HIGH);
    analogWrite(PWMB, -speed);

  } 
  else {

    digitalWrite(BIN1, LOW);
    digitalWrite(BIN2, LOW);
    analogWrite(PWMB, 0);
  }
}


// ================= BOTH MOTORS =================

void motors(int leftSpeed, int rightSpeed) {

  motorA(leftSpeed);
  motorB(rightSpeed);
}


// ================= MOTORS ON =================

void motorsOn() {

  // Your working forward direction
  motors(43, -40);
}


// ================= MOTORS OFF =================

void motorsOff() {

  motors(0, 0);
}


// ================= MATRIX OFF =================

void matrixOff() {

  for (int i = 0; i < 8; i++) {
    digitalWrite(rowPins[i], LOW);
  }

  for (int i = 0; i < 8; i++) {
    digitalWrite(colPins[i], HIGH);
  }
}


// ================= MATRIX SCAN =================

void scanMatrix() {

  // Turn current row OFF
  digitalWrite(rowPins[currentRow], LOW);

  // Turn all columns ON
  for (int col = 0; col < 8; col++) {
    digitalWrite(colPins[col], LOW);
  }

  // Next row
  currentRow++;

  if (currentRow >= 8) {
    currentRow = 0;
  }

  // Turn new row ON
  digitalWrite(rowPins[currentRow], HIGH);
}


// ================= SETUP =================

void setup() {

  // --------------------------
  // Motor pins
  // --------------------------

  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(PWMA, OUTPUT);

  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);
  pinMode(PWMB, OUTPUT);


  // --------------------------
  // Matrix pins
  // --------------------------

  for (int i = 0; i < 8; i++) {
    pinMode(rowPins[i], OUTPUT);
    pinMode(colPins[i], OUTPUT);
  }


  // ==========================================================
  // DEFAULT STATE = 00
  // ==========================================================

  ledState = false;
  motorState = false;

  matrixOff();
  motorsOff();


  // --------------------------
  // Serial
  // --------------------------

  Serial.begin(115200);

  Serial.println();
  Serial.println("==============================");
  Serial.println("ESP32-P4 ROBOT");
  Serial.println("==============================");
  Serial.println("DEFAULT STATE: 00");
  Serial.println();
  Serial.println("00 = LED OFF / MOTOR OFF");
  Serial.println("01 = LED OFF / MOTOR ON");
  Serial.println("10 = LED ON  / MOTOR OFF");
  Serial.println("11 = LED ON  / MOTOR ON");
}


// ================= LOOP =================

void loop() {

  // ==========================================================
  // SERIAL INPUT
  // ==========================================================

  if (Serial.available()) {

    String command = Serial.readStringUntil('\n');

    command.trim();


    // --------------------------
    // 00
    // --------------------------

    if (command == "00") {

      ledState = false;
      motorState = false;

      matrixOff();
      motorsOff();

      Serial.println("00 -> LED OFF | MOTOR OFF");
    }


    // --------------------------
    // 01
    // --------------------------

    else if (command == "01") {

      ledState = false;
      motorState = true;

      matrixOff();
      motorsOn();

      Serial.println("01 -> LED OFF | MOTOR ON");
    }


    // --------------------------
    // 10
    // --------------------------

    else if (command == "10") {

      ledState = true;
      motorState = false;

      motorsOff();

      Serial.println("10 -> LED ON | MOTOR OFF");
    }


    // --------------------------
    // 11
    // --------------------------

    else if (command == "11") {

      ledState = true;
      motorState = true;

      motorsOn();

      Serial.println("11 -> LED ON | MOTOR ON");
    }


    // --------------------------
    // INVALID
    // --------------------------

    else {

      Serial.println("Invalid command!");
      Serial.println("Use 00, 01, 10 or 11");
    }
  }


  // ==========================================================
  // MATRIX REFRESH
  // ==========================================================

  if (ledState) {

    if (millis() - lastScan >= 1) {

      lastScan = millis();

      scanMatrix();
    }

  } 
  else {

    matrixOff();
  }
}
