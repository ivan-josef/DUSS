#!/usr/bin/env python3
"""
Executa segmentação, medição e classificação visual para uma imagem.

Uso:
    python avaliar_completo.py imagem.jpg formulario_enfermeira.json \
        --classificador modelos/classificador_aprovado.pth
"""

import argparse
import json
from pathlib import Path

import cv2
from rfdetr import RFDETRSegMedium

import pipeline_ulcera
from classificador_caracteristicas import (
    ClassificadorCaracteristicas,
    recortar_lesao,
)


def obter_medidas_enfermeira(formulario):
    origem = formulario.get("medidas_manuais", formulario)
    comprimento = origem.get("comprimento_cm")
    largura = origem.get("largura_cm")
    if comprimento is None or largura is None:
        return None
    return {
        "comprimento_cm": float(comprimento),
        "largura_cm": float(largura),
    }


def salvar_resultado(resultado, caminho_imagem, pastas):
    texto = json.dumps(resultado, indent=2, ensure_ascii=False)
    (pastas["json"] / f"{caminho_imagem.stem}.json").write_text(
        texto, encoding="utf-8"
    )
    (pipeline_ulcera.OUTPUT_DIR / "resultado.json").write_text(
        texto, encoding="utf-8"
    )


def executar(args):
    caminho_imagem = Path(args.imagem)
    caminho_formulario = Path(args.formulario)
    if not caminho_imagem.is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")
    if not caminho_formulario.is_file():
        raise FileNotFoundError(
            f"Formulário não encontrado: {caminho_formulario}"
        )

    formulario = json.loads(caminho_formulario.read_text(encoding="utf-8"))
    pipeline_ulcera.OUTPUT_DIR = Path(args.output_dir)
    if args.pixels_por_cm is not None:
        pipeline_ulcera.PIXELS_POR_CM = args.pixels_por_cm
    pastas = pipeline_ulcera.criar_pastas()
    pasta_recortes = pipeline_ulcera.OUTPUT_DIR / "recortes_classificador"
    pasta_recortes.mkdir(parents=True, exist_ok=True)

    print("Carregando o segmentador...")
    segmentador = RFDETRSegMedium(pretrain_weights=args.modelo_segmentacao)
    resultado = pipeline_ulcera.processar(
        segmentador,
        caminho_imagem,
        pastas,
        pipeline_ulcera.obter_escala(),
        obter_medidas_enfermeira(formulario),
    )
    if resultado is None:
        raise RuntimeError("Não foi possível obter uma máscara válida.")

    imagem = cv2.imread(str(caminho_imagem))
    caminho_mascara = (
        pastas["mascaras_processadas"] / f"{caminho_imagem.stem}.png"
    )
    mascara = cv2.imread(str(caminho_mascara), cv2.IMREAD_GRAYSCALE)
    recorte = recortar_lesao(imagem, mascara)
    caminho_recorte = pasta_recortes / f"{caminho_imagem.stem}.png"
    if not cv2.imwrite(str(caminho_recorte), recorte):
        raise OSError(f"Não foi possível salvar o recorte: {caminho_recorte}")

    previsao = {
        "status": "nao_executado",
        "motivo": "Nenhum checkpoint aprovado foi informado.",
    }
    if args.classificador:
        print("Carregando o classificador...")
        classificador = ClassificadorCaracteristicas(
            args.classificador, args.device
        )
        previsao = {"status": "executado", **classificador.prever(recorte)}

    resultado["patient_id"] = (
        formulario.get("patient_id") or formulario.get("paciente_id")
    )
    resultado["visit_id"] = (
        formulario.get("visit_id") or formulario.get("consulta_id")
    )
    resultado["avaliacao_enfermeira"] = formulario
    resultado["dados_enfermeira"] = formulario
    resultado["previsao_classificador"] = previsao
    resultado["status_revisao_medica"] = "pendente"
    resultado["confirmacao_medica"] = None
    resultado["recorte_classificador"] = str(caminho_recorte.resolve())
    resultado["arquivos"]["recorte_classificador"] = (
        f"recortes_classificador/{caminho_recorte.name}"
    )
    salvar_resultado(resultado, caminho_imagem, pastas)

    print(f"Concluído. Resultados em: {pipeline_ulcera.OUTPUT_DIR}")
    return resultado


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("imagem")
    parser.add_argument("formulario")
    parser.add_argument(
        "--classificador",
        help="Checkpoint aprovado. Omita antes de existir a primeira versão.",
    )
    parser.add_argument(
        "--modelo-segmentacao",
        default=pipeline_ulcera.MODEL_PATH,
    )
    parser.add_argument(
        "--output-dir",
        default=str(pipeline_ulcera.OUTPUT_DIR),
    )
    parser.add_argument("--pixels-por-cm", type=float)
    parser.add_argument("--device", help="Ex.: cuda ou cpu")
    executar(parser.parse_args())


if __name__ == "__main__":
    main()
