#!/usr/bin/env python3
"""Treina e avalia o classificador multirrótulo em lotes versionados."""

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from classificador_caracteristicas import (
    criar_modelo,
    transformar_avaliacao,
)
from config_classificador import CLASSES, INPUT_SIZE


def fixar_semente(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


def ler_csv(caminho_csv):
    registros = []
    with Path(caminho_csv).open(newline="", encoding="utf-8") as arquivo:
        for linha in csv.DictReader(arquivo):
            if not linha.get("patient_id") or not linha.get("imagem"):
                continue
            rotulos = []
            for nome in CLASSES:
                valor = str(linha.get(nome, "")).strip()
                rotulos.append(float(valor) if valor in {"0", "1"} else -1.0)
            registros.append(
                {
                    "patient_id": linha["patient_id"],
                    "visit_id": linha.get("visit_id", ""),
                    "imagem": linha["imagem"],
                    "rotulos": rotulos,
                }
            )
    if not registros:
        raise ValueError("O CSV não contém registros válidos.")
    return registros


def dividir_por_paciente(registros, val_ratio, test_ratio, semente):
    pacientes = sorted({item["patient_id"] for item in registros})
    if len(pacientes) < 3:
        raise ValueError(
            "São necessários ao menos 3 pacientes distintos para separar "
            "treino, validação e teste sem vazamento entre pacientes."
        )

    random.Random(semente).shuffle(pacientes)
    n_test = max(1, round(len(pacientes) * test_ratio))
    n_val = max(1, round(len(pacientes) * val_ratio))
    while n_test + n_val >= len(pacientes):
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("Não foi possível criar três partições.")

    pacientes_test = set(pacientes[:n_test])
    pacientes_val = set(pacientes[n_test : n_test + n_val])
    pacientes_treino = set(pacientes[n_test + n_val :])

    def selecionar(grupo):
        return [item for item in registros if item["patient_id"] in grupo]

    return (
        selecionar(pacientes_treino),
        selecionar(pacientes_val),
        selecionar(pacientes_test),
    )


class DatasetLesoes(Dataset):
    def __init__(self, registros, transform):
        self.registros = registros
        self.transform = transform

    def __len__(self):
        return len(self.registros)

    def __getitem__(self, indice):
        registro = self.registros[indice]
        imagem = Image.open(registro["imagem"]).convert("RGB")
        rotulos = torch.tensor(registro["rotulos"], dtype=torch.float32)
        validos = (rotulos >= 0).float()
        rotulos = torch.clamp(rotulos, min=0)
        return self.transform(imagem), rotulos, validos


def transformacao_treino():
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(12),
            transforms.ColorJitter(
                brightness=0.12, contrast=0.12, saturation=0.08
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def calcular_pos_weight(registros, device):
    pesos = []
    for indice in range(len(CLASSES)):
        conhecidos = [
            item["rotulos"][indice]
            for item in registros
            if item["rotulos"][indice] >= 0
        ]
        positivos = sum(valor == 1 for valor in conhecidos)
        negativos = sum(valor == 0 for valor in conhecidos)
        if positivos == 0 or negativos == 0:
            pesos.append(1.0)
        else:
            pesos.append(negativos / positivos)
    return torch.tensor(pesos, dtype=torch.float32, device=device)


def perda_mascarada(logits, alvos, validos, pos_weight):
    perdas = F.binary_cross_entropy_with_logits(
        logits, alvos, pos_weight=pos_weight, reduction="none"
    )
    return (perdas * validos).sum() / validos.sum().clamp(min=1)


def executar_epoca(modelo, loader, device, pos_weight, otimizador=None):
    treinamento = otimizador is not None
    modelo.train(treinamento)
    soma_perda = 0.0
    lotes = 0

    for imagens, alvos, validos in loader:
        imagens = imagens.to(device)
        alvos = alvos.to(device)
        validos = validos.to(device)
        if treinamento:
            otimizador.zero_grad(set_to_none=True)
        logits = modelo(imagens)
        perda = perda_mascarada(logits, alvos, validos, pos_weight)
        if treinamento:
            perda.backward()
            otimizador.step()
        soma_perda += float(perda.item())
        lotes += 1
    return soma_perda / max(lotes, 1)


@torch.inference_mode()
def coletar_previsoes(modelo, loader, device):
    modelo.eval()
    probabilidades, alvos, validos = [], [], []
    for imagens, lote_alvos, lote_validos in loader:
        logits = modelo(imagens.to(device))
        probabilidades.append(torch.sigmoid(logits).cpu().numpy())
        alvos.append(lote_alvos.numpy())
        validos.append(lote_validos.numpy())
    return (
        np.concatenate(probabilidades),
        np.concatenate(alvos),
        np.concatenate(validos).astype(bool),
    )


def metricas_binarias(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "suporte_positivo": int(np.sum(y_true == 1)),
        "suporte_total": int(len(y_true)),
    }


def ajustar_thresholds(probabilidades, alvos, validos):
    thresholds = {}
    for indice, nome in enumerate(CLASSES):
        mascara = validos[:, indice]
        y_true = alvos[mascara, indice].astype(int)
        y_prob = probabilidades[mascara, indice]
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            thresholds[nome] = 0.5
            continue
        melhor_threshold, melhor_f1 = 0.5, -1.0
        for threshold in np.arange(0.10, 0.91, 0.05):
            f1 = metricas_binarias(
                y_true, (y_prob >= threshold).astype(int)
            )["f1"]
            if f1 > melhor_f1:
                melhor_threshold, melhor_f1 = float(threshold), f1
        thresholds[nome] = round(melhor_threshold, 2)
    return thresholds


def avaliar(probabilidades, alvos, validos, thresholds):
    resultado = {}
    for indice, nome in enumerate(CLASSES):
        mascara = validos[:, indice]
        y_true = alvos[mascara, indice].astype(int)
        y_pred = (
            probabilidades[mascara, indice] >= thresholds[nome]
        ).astype(int)
        resultado[nome] = metricas_binarias(y_true, y_pred)
        resultado[nome]["threshold"] = thresholds[nome]
    resultado["f1_macro"] = round(
        float(np.mean([resultado[nome]["f1"] for nome in CLASSES])), 4
    )
    return resultado


def treinar(args):
    fixar_semente(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    registros = ler_csv(args.csv)
    treino, validacao, teste = dividir_por_paciente(
        registros, args.val_ratio, args.test_ratio, args.seed
    )
    print(
        f"Dispositivo: {device} | treino={len(treino)}, "
        f"validação={len(validacao)}, teste={len(teste)}"
    )

    loaders = {
        "treino": DataLoader(
            DatasetLesoes(treino, transformacao_treino()),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
        ),
        "validacao": DataLoader(
            DatasetLesoes(validacao, transformar_avaliacao()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        ),
        "teste": DataLoader(
            DatasetLesoes(teste, transformar_avaliacao()),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        ),
    }

    modelo = criar_modelo(
        len(CLASSES), pretrained=not args.sem_pretreino and not args.resume
    )
    if args.resume:
        anterior = torch.load(args.resume, map_location="cpu")
        if anterior.get("classes", CLASSES) != CLASSES:
            raise ValueError("O checkpoint anterior possui classes diferentes.")
        modelo.load_state_dict(anterior["model_state_dict"])
        print(f"Retreinamento iniciado a partir de: {args.resume}")

    modelo.to(device)
    pos_weight = calcular_pos_weight(treino, device)
    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=args.lr, weight_decay=1e-4
    )

    melhor_estado = copy.deepcopy(modelo.state_dict())
    melhor_val = float("inf")
    paciencia = 0
    historico = []
    for epoca in range(1, args.epochs + 1):
        perda_treino = executar_epoca(
            modelo, loaders["treino"], device, pos_weight, otimizador
        )
        perda_val = executar_epoca(
            modelo, loaders["validacao"], device, pos_weight
        )
        historico.append(
            {
                "epoca": epoca,
                "perda_treino": round(perda_treino, 6),
                "perda_validacao": round(perda_val, 6),
            }
        )
        print(
            f"Época {epoca:03d}: treino={perda_treino:.4f} "
            f"validação={perda_val:.4f}"
        )
        if perda_val < melhor_val:
            melhor_val = perda_val
            melhor_estado = copy.deepcopy(modelo.state_dict())
            paciencia = 0
        else:
            paciencia += 1
            if paciencia >= args.early_stopping:
                print("Early stopping acionado.")
                break

    modelo.load_state_dict(melhor_estado)
    prob_val, y_val, mask_val = coletar_previsoes(
        modelo, loaders["validacao"], device
    )
    thresholds = ajustar_thresholds(prob_val, y_val, mask_val)
    prob_test, y_test, mask_test = coletar_previsoes(
        modelo, loaders["teste"], device
    )
    metricas = avaliar(prob_test, y_test, mask_test, thresholds)

    saida = Path(args.output)
    saida.parent.mkdir(parents=True, exist_ok=True)
    pacote = {
        "model_state_dict": modelo.cpu().state_dict(),
        "architecture": "efficientnet_b0",
        "classes": CLASSES,
        "thresholds": thresholds,
        "version": args.version,
        "input_size": INPUT_SIZE,
        "metrics_test": metricas,
        "split": {
            "registros_treino": len(treino),
            "registros_validacao": len(validacao),
            "registros_teste": len(teste),
            "seed": args.seed,
        },
    }
    torch.save(pacote, saida)

    relatorio = {
        "checkpoint": str(saida),
        "version": args.version,
        "melhor_perda_validacao": round(melhor_val, 6),
        "thresholds": thresholds,
        "metricas_teste": metricas,
        "historico": historico,
    }
    saida.with_suffix(".json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Checkpoint candidato salvo em: {saida}")
    print(json.dumps(metricas, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV exportado pelo sistema.")
    parser.add_argument(
        "--output", required=True, help="Ex.: modelos/classificador_v1.pth"
    )
    parser.add_argument("--version", required=True, help="Ex.: 1.0.0")
    parser.add_argument("--resume", help="Checkpoint aprovado anterior.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--early-stopping", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", help="Ex.: cuda, cuda:0 ou cpu")
    parser.add_argument(
        "--sem-pretreino",
        action="store_true",
        help="Não carregar pesos ImageNet na primeira versão.",
    )
    args = parser.parse_args()
    treinar(args)


if __name__ == "__main__":
    main()
