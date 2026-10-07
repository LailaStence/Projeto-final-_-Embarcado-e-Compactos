/*
 * Reconhecimento de Gestos com IA Embarcada - ESP32-S3
 * Pipeline: sensor MPU6050 -> janela 128 amostras @ 50 Hz ->
 *           extracao de features -> normalizacao int8 -> MLP int8 -> gesto
 * Funciona no hardware real ou no simulador Wokwi.
 */
#include <Wire.h>
#include "gesture_model.h"

#define SDA_PIN 8
#define SCL_PIN 9
#define MPU_ADDR 0x68
#define WIN 128
#define FS_HZ 50

static const char *LABELS[N_CLASSES] = {"repouso", "caminhada", "sacudida", "queda"};

float buf[WIN][3];
int idx = 0;

void mpu_init() {
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B); Wire.write(0);   // desperta o MPU6050
  Wire.endTransmission();
}

void mpu_read(float *ax, float *ay, float *az) {   // em g
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); Wire.endTransmission();
  Wire.requestFrom(MPU_ADDR, 6);
  int16_t x = (Wire.read() << 8) | Wire.read();
  int16_t y = (Wire.read() << 8) | Wire.read();
  int16_t z = (Wire.read() << 8) | Wire.read();
  *ax = x / 16384.0f; *ay = y / 16384.0f; *az = z / 16384.0f;
}

// Caso nao haja sensor: simula uma caminhada para demonstracao no Wokwi
void sim_read(float *ax, float *ay, float *az, unsigned long t) {
  float s = sinf(2.0f * PI * 1.5f * t / 1000.0f);
  *ax = 0.3f * s + 0.02f * random(-100, 100) / 100.0f;
  *ay = 0.1f * s;
  *az = 1.0f + 0.1f * cosf(2.0f * PI * 1.5f * t / 1000.0f);
}

// Extrai as 15 features da janela e quantiza para int8
void extract_features(const float w[WIN][3], int8_t *out) {
  float f[N_FEATURES] = {0};
  for (int a = 0; a < 3; a++) {
    float m = 0, mn = 1e9, mx = -1e9, e = 0;
    for (int i = 0; i < WIN; i++) {
      float v = w[i][a]; m += v; e += v * v;
      if (v < mn) mn = v; if (v > mx) mx = v;
    }
    m /= WIN; e = sqrtf(e / WIN);
    float sd = 0;
    for (int i = 0; i < WIN; i++) { float d = w[i][a] - m; sd += d * d; }
    sd = sqrtf(sd / WIN);
    f[a] = m; f[3 + a] = sd; f[6 + a] = mx; f[9 + a] = mn; f[12 + a] = e;
  }
  // normalizacao fixa + quantizacao simetrica int8
  for (int i = 0; i < N_FEATURES; i++) {
    int32_t v = ((int32_t)f[i] - MU_INT8[i]) * INV_STD_INT8[i];
    v = v < -127 ? -127 : (v > 127 ? 127 : v);
    out[i] = (int8_t)v;
  }
}

void setup() {
  Serial.begin(115200);
  mpu_init();
  Serial.println("IA Embarcada - Reconhecimento de Gestos (ESP32-S3)");
}

void loop() {
  unsigned long t0 = millis();
  float ax, ay, az;
  mpu_read(&ax, &ay, &az);          // troque por sim_read(&ax,&ay,&az,t0) no Wokwi sem MPU
  buf[idx][0] = ax; buf[idx][1] = ay; buf[idx][2] = az;
  idx++;
  if (idx >= WIN) {                 // janela completa: inferencia
    idx = 0;
    int8_t x8[N_FEATURES];
    extract_features(buf, x8);
    int32_t logits[N_CLASSES];
    gesture_inference(x8, logits);
    int pred = gesture_argmax(logits);
    Serial.printf("Gesto: %-10s | logits: %ld %ld %ld %ld\n",
                  LABELS[pred], logits[0], logits[1], logits[2], logits[3]);
  }
  // amostragem fixa em 50 Hz
  long dt = millis() - t0;
  if (dt < 1000 / FS_HZ) delay(1000 / FS_HZ - dt);
}
