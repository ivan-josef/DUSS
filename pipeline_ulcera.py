#!/usr/bin/env python3
"""
Segmentação e medição de uma úlcera com RF-DETR-Seg-Medium.

Uso:
    python pipeline_ulcera.py test/imagem.jpg 3.0 2.0

Os dois últimos valores são o comprimento e a largura informados
pela enfermeira, em centímetros.
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRSegMedium


# Ajuste estas configurações antes de executar.
MODEL_PATH = "modelos/checkpoint_best_total.pth"
DEFAULT_IMAGE_PATH = Path("test/013_jpg.rf.106421025ded2c81fa655dda42d005f0.jpg")
OUTPUT_DIR = Path("resultados_ulceras")
THRESHOLD = 0.5
MIN_REGION_AREA = 100

# Informe a escala diretamente...
PIXELS_POR_CM = 125.0

# ...ou use None acima e informe as dimensões do marcador.
LARGURA_MARCADOR_PX = None
LARGURA_MARCADOR_CM = None

def obter_escala():
    if PIXELS_POR_CM is not None:
        return float(PIXELS_POR_CM)
    if LARGURA_MARCADOR_PX and LARGURA_MARCADOR_CM:
        return LARGURA_MARCADOR_PX / LARGURA_MARCADOR_CM
    return None


def criar_pastas():
    nomes = [
        "originais",
        "mascaras_originais",
        "mascaras_processadas",
        "overlays",
        "todas_deteccoes",
        "json",
    ]
    pastas = {nome: OUTPUT_DIR / nome for nome in nomes}
    for pasta in pastas.values():
        pasta.mkdir(parents=True, exist_ok=True)
    return pastas


def pos_processar(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Preenchimento de buracos.
    preenchida = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    flood = preenchida.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    preenchida |= cv2.bitwise_not(flood)
    mask = preenchida[1:-1, 1:-1]

    # Remoção de regiões pequenas e seleção do maior componente.
    total, rotulos, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    componentes = [
        (rotulo, stats[rotulo, cv2.CC_STAT_AREA])
        for rotulo in range(1, total)
        if stats[rotulo, cv2.CC_STAT_AREA] >= MIN_REGION_AREA
    ]
    if not componentes:
        return np.zeros_like(mask)

    maior = max(componentes, key=lambda item: item[1])[0]
    return np.where(rotulos == maior, 255, 0).astype(np.uint8)


def segmentar(modelo, imagem_pil):
    deteccoes = modelo.predict(imagem_pil, threshold=THRESHOLD)
    if deteccoes.mask is None or len(deteccoes.mask) == 0:
        return None, None, [], []

    largura, altura = imagem_pil.size
    mascaras = []
    for mask_detectada in deteccoes.mask:
        mask = (np.asarray(mask_detectada) > 0).astype(np.uint8) * 255
        if mask.shape != (altura, largura):
            mask = cv2.resize(
                mask, (largura, altura), interpolation=cv2.INTER_NEAREST
            )
        mascaras.append(mask)

    confiancas = [float(valor) for valor in deteccoes.confidence]
    indice = int(np.argmax(deteccoes.confidence))
    return mascaras[indice], confiancas[indice], mascaras, confiancas


def medir(mask, pixels_por_cm):
    contornos, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contornos:
        return None, None, None

    contorno = max(contornos, key=cv2.contourArea)
    retangulo = cv2.minAreaRect(contorno)
    lado_a, lado_b = retangulo[1]
    caixa = cv2.boxPoints(retangulo).astype(np.int32)

    area_px = int(cv2.countNonZero(mask))
    perimetro_px = float(cv2.arcLength(contorno, True))
    comprimento_px = float(max(lado_a, lado_b))
    largura_px = float(min(lado_a, lado_b))

    medidas = {
        "area_px": area_px,
        "perimetro_px": round(perimetro_px, 2),
        "comprimento_px": round(comprimento_px, 2),
        "largura_px": round(largura_px, 2),
        "pixels_por_cm": pixels_por_cm,
        "area_cm2": None,
        "perimetro_cm": None,
        "comprimento_cm": None,
        "largura_cm": None,
    }
    if pixels_por_cm:
        medidas.update(
            area_cm2=round(area_px / pixels_por_cm**2, 2),
            perimetro_cm=round(perimetro_px / pixels_por_cm, 2),
            comprimento_cm=round(comprimento_px / pixels_por_cm, 2),
            largura_cm=round(largura_px / pixels_por_cm, 2),
        )
    return medidas, contorno, caixa


def criar_overlay(imagem, mask, contorno, caixa, confianca, medidas):
    cor = np.zeros_like(imagem)
    cor[mask > 0] = (0, 0, 255)
    resultado = cv2.addWeighted(imagem, 1.0, cor, 0.35, 0)
    cv2.drawContours(resultado, [contorno], -1, (0, 255, 0), 2)
    cv2.polylines(resultado, [caixa], True, (255, 180, 0), 2)

    texto = f"Confianca: {confianca:.2f}"
    if medidas["area_cm2"] is not None:
        texto += (
            f" | Area: {medidas['area_cm2']:.2f} cm2"
            f" | C: {medidas['comprimento_cm']:.2f} cm"
            f" | L: {medidas['largura_cm']:.2f} cm"
        )
    cv2.putText(
        resultado,
        texto,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return resultado


def criar_overlay_todas(imagem, mascaras, confiancas):
    """Sobrepõe todas as máscaras detectadas na imagem original."""
    cores = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
    ]
    resultado = imagem.copy()

    for indice, (mask, confianca) in enumerate(zip(mascaras, confiancas)):
        cor = cores[indice % len(cores)]
        camada = resultado.copy()
        camada[mask > 0] = cor
        resultado = cv2.addWeighted(camada, 0.35, resultado, 0.65, 0)

        contornos, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(resultado, contornos, -1, cor, 2)
        if contornos:
            x, y, _, _ = cv2.boundingRect(max(contornos, key=cv2.contourArea))
            cv2.putText(
                resultado,
                f"{indice + 1}: {confianca:.2f}",
                (x, max(20, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return resultado


def processar(modelo, caminho, pastas, escala, dados_enfermeira=None):
    imagem_pil = Image.open(caminho).convert("RGB")
    imagem = cv2.cvtColor(np.array(imagem_pil), cv2.COLOR_RGB2BGR)
    mask_original, confianca, mascaras, confiancas = segmentar(
        modelo, imagem_pil
    )
    if mask_original is None:
        print("  Nenhuma úlcera detectada.")
        return None

    mask_processada = pos_processar(mask_original)
    medidas, contorno, caixa = medir(mask_processada, escala)
    if medidas is None:
        print("  Máscara vazia após o pós-processamento.")
        return None

    comparacao = None
    if dados_enfermeira and medidas["comprimento_cm"] is not None:
        comparacao = {
            "comprimento_informado_cm": dados_enfermeira["comprimento_cm"],
            "largura_informada_cm": dados_enfermeira["largura_cm"],
            "diferenca_comprimento_cm": round(
                abs(
                    medidas["comprimento_cm"]
                    - dados_enfermeira["comprimento_cm"]
                ),
                2,
            ),
            "diferenca_largura_cm": round(
                abs(medidas["largura_cm"] - dados_enfermeira["largura_cm"]),
                2,
            ),
        }

    nome_png = caminho.stem + ".png"
    overlay = criar_overlay(
        imagem, mask_processada, contorno, caixa, confianca, medidas
    )
    overlay_todas = criar_overlay_todas(imagem, mascaras, confiancas)
    shutil.copy2(caminho, pastas["originais"] / caminho.name)
    cv2.imwrite(str(pastas["mascaras_originais"] / nome_png), mask_original)
    cv2.imwrite(
        str(pastas["mascaras_processadas"] / nome_png), mask_processada
    )
    cv2.imwrite(str(pastas["overlays"] / nome_png), overlay)
    cv2.imwrite(
        str(pastas["todas_deteccoes"] / nome_png), overlay_todas
    )

    resultado = {
        "imagem": caminho.name,
        "modelo": "RF-DETR-Seg-Medium",
        "confianca": round(confianca, 4),
        "deteccoes": [
            {"numero": indice + 1, "confianca": round(valor, 4)}
            for indice, valor in enumerate(confiancas)
        ],
        "dados_enfermeira": dados_enfermeira,
        "medidas": medidas,
        "comparacao_enfermeira": comparacao,
        "arquivos": {
            "original": f"originais/{caminho.name}",
            "mascara_original": f"mascaras_originais/{nome_png}",
            "mascara_processada": f"mascaras_processadas/{nome_png}",
            "overlay": f"overlays/{nome_png}",
            "todas_deteccoes": f"todas_deteccoes/{nome_png}",
        },
    }
    with open(
        pastas["json"] / f"{caminho.stem}.json", "w", encoding="utf-8"
    ) as arquivo:
        json.dump(resultado, arquivo, indent=2, ensure_ascii=False)
    return resultado


def main():
    pastas = criar_pastas()
    escala = obter_escala()
    if len(sys.argv) not in (1, 2, 4):
        raise RuntimeError(
            "Uso: python pipeline_ulcera.py IMAGEM [COMPRIMENTO_CM LARGURA_CM]"
        )

    caminho = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    if not caminho.is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {caminho}")

    dados_enfermeira = None
    if len(sys.argv) == 4:
        dados_enfermeira = {
            "comprimento_cm": float(sys.argv[2]),
            "largura_cm": float(sys.argv[3]),
        }

    print("Carregando o modelo...")
    modelo = RFDETRSegMedium(pretrain_weights=MODEL_PATH)
    print(f"Processando: {caminho.name}")
    resultado = processar(
        modelo, caminho, pastas, escala, dados_enfermeira
    )
    if resultado is None:
        raise RuntimeError("Não foi possível obter uma máscara válida.")

    with open(OUTPUT_DIR / "resultado.json", "w", encoding="utf-8") as arquivo:
        json.dump(resultado, arquivo, indent=2, ensure_ascii=False)
    print(f"Concluído. Resultados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
