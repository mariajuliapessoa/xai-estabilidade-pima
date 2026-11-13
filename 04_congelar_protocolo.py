"""Etapa 4 — Congela o protocolo: registra os hashes da configuração, do código
e dos dados. Os scripts que usam o conjunto de teste (05 e 06) só rodam se esses
arquivos não tiverem sido alterados depois do congelamento.
Uso: python 04_congelar_protocolo.py --confirmo"""
import sys

import xai_estab as xe

cfg = xe.carregar_config()
if "--confirmo" not in sys.argv:
    print("Decisões que devem estar confirmadas antes de congelar:")
    for d in cfg["decisoes_pendentes"]:
        print(" -", d)
    print(f"\nValor atual de M: {cfg['repeticoes']['M']} (confira com intermediarios/piloto.json)")
    sys.exit("Depois de confirmar e ajustar config.json, rode: python 04_congelar_protocolo.py --confirmo")

xe.salvar_json({"data": xe.agora(), "hashes": xe.hashes_protocolo(cfg), "versoes": xe.versoes(),
                "config": cfg}, "protocolo_congelado.json")
print("Protocolo congelado em protocolo_congelado.json. Não altere config.json nem o código daqui em diante.")
