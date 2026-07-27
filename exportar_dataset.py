#!/usr/bin/env python3
"""
Transforma avaliações confirmadas pelo médico em um CSV de treinamento.

Somente registros explicitamente confirmados são exportados. Valores
"não avaliado" permanecem vazios e não são convertidos em ausentes.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

from config_classificador import CLASSES


VALORES = {
    "presente": "1",
    "sim": "1",
    "true": "1",
    "1": "1",
    "ausente": "0",
    "nao": "0",
    "não": "0",
    "false": "0",
    "0": "0",
}


def obter_confirmacao(registro):
    for chave in ("confirmacao_medica", "avaliacao_medico"):
        valor = registro.get(chave)
        if isinstance(valor, dict):
            return valor
    return {}


def esta_confirmado(registro, confirmacao):
    status = str(registro.get("status_revisao_medica", "")).lower()
    status_confirmacao = str(confirmacao.get("status", "")).lower()
    return bool(registro.get("confirmado_pelo_medico")) or status in {
        "confirmado",
        "confirmado_pelo_medico",
    } or status_confirmacao in {"confirmado", "concluido", "concluído"}


def obter_rotulos(confirmacao):
    origem = confirmacao.get("caracteristicas_visuais", confirmacao)
    rotulos = {}
    for nome in CLASSES:
        valor = origem.get(nome, "")
        if isinstance(valor, bool):
            rotulos[nome] = "1" if valor else "0"
        else:
            rotulos[nome] = VALORES.get(str(valor).strip().lower(), "")
    return rotulos


def obter_recorte(registro, arquivo_json):
    recorte = registro.get("recorte_classificador")
    if not recorte:
        recorte = registro.get("arquivos", {}).get("recorte_classificador")
    if not recorte:
        return None

    caminho = Path(recorte)
    if caminho.is_absolute():
        return caminho

    candidatos = [
        arquivo_json.parent / caminho,
        arquivo_json.parent.parent / caminho,
    ]
    return next((item.resolve() for item in candidatos if item.is_file()), None)


def exportar(diretorio_registros, arquivo_saida, copiar_imagens=True):
    diretorio_registros = Path(diretorio_registros)
    arquivo_saida = Path(arquivo_saida)
    arquivo_saida.parent.mkdir(parents=True, exist_ok=True)
    pasta_imagens = arquivo_saida.parent / f"{arquivo_saida.stem}_imagens"
    if copiar_imagens:
        pasta_imagens.mkdir(parents=True, exist_ok=True)

    linhas = []
    ignorados = []
    for arquivo_json in sorted(diretorio_registros.rglob("*.json")):
        try:
            registro = json.loads(arquivo_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as erro:
            ignorados.append(f"{arquivo_json}: JSON inválido ({erro})")
            continue

        confirmacao = obter_confirmacao(registro)
        if not esta_confirmado(registro, confirmacao):
            continue

        paciente = str(
            registro.get("patient_id")
            or registro.get("paciente_id")
            or registro.get("avaliacao_enfermeira", {}).get("paciente_id")
            or ""
        ).strip()
        visita = str(
            registro.get("visit_id")
            or registro.get("consulta_id")
            or registro.get("avaliacao_enfermeira", {}).get("consulta_id")
            or arquivo_json.stem
        ).strip()
        recorte = obter_recorte(registro, arquivo_json)
        if not paciente or recorte is None or not recorte.is_file():
            ignorados.append(
                f"{arquivo_json}: paciente_id ou recorte ausente."
            )
            continue

        if copiar_imagens:
            destino = pasta_imagens / (
                f"{paciente}_{visita}_{recorte.name}".replace(" ", "_")
            )
            shutil.copy2(recorte, destino)
            caminho_imagem = destino.resolve()
        else:
            caminho_imagem = recorte.resolve()

        linha = {
            "patient_id": paciente,
            "visit_id": visita,
            "imagem": str(caminho_imagem),
        }
        linha.update(obter_rotulos(confirmacao))
        if all(linha[nome] == "" for nome in CLASSES):
            ignorados.append(f"{arquivo_json}: nenhum rótulo médico utilizável.")
            continue
        linhas.append(linha)

    campos = ["patient_id", "visit_id", "imagem", *CLASSES]
    with arquivo_saida.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)

    print(f"Registros exportados: {len(linhas)}")
    print(f"CSV criado em: {arquivo_saida}")
    if ignorados:
        log = arquivo_saida.with_suffix(".ignorados.txt")
        log.write_text("\n".join(ignorados), encoding="utf-8")
        print(f"Registros ignorados: {len(ignorados)} ({log})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "diretorio_registros",
        help="Pasta que contém os JSONs revisados pelo médico.",
    )
    parser.add_argument("arquivo_saida", help="Ex.: dataset/rotulos.csv")
    parser.add_argument(
        "--nao-copiar-imagens",
        action="store_true",
        help="Use os recortes nos caminhos originais em vez de copiá-los.",
    )
    args = parser.parse_args()
    exportar(
        args.diretorio_registros,
        args.arquivo_saida,
        copiar_imagens=not args.nao_copiar_imagens,
    )


if __name__ == "__main__":
    main()
