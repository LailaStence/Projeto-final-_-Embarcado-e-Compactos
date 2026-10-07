# IA Embarcada - Reconhecimento de Gestos (ESP32-S3)

Pipeline completa de TinyML: sensor -> features -> MLP int8 -> classe.

## Estrutura
- `firmware/gesture_recognition.ino` - sketch Arduino (ESP32-S3): leitura MPU6050, features, inferência int8
- `firmware/gesture_model.h` - pesos quantizados int8 + função de inferência inline
- `firmware/diagram.json` - esquema para simulação no Wokwi
- `train/modelo.pkl` - pipeline scikit-learn treinado (15->16->4)
- `Apresentacao_Projeto_Final_IA_Embarcada.pdf` - slides da apresentação

## Git flow
Branches: `main` (estável), `develop`, `feature/*`, `release/*`.
Cada integrante commita em branches de feature e abre PR para `develop`.

## Execução
1. Wokwi: abrir `firmware/` no Wokwi e rodar (funciona com MPU6050 ou com sensor simulado).
2. Hardware real: compilar o `.ino` com Arduino IDE (placa ESP32-S3 DevKit).
