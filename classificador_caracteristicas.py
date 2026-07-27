#!/usr/bin/env python3
"""Inferência multirrótulo das características visuais da lesão."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from config_classificador import CLASSES, INPUT_SIZE


def criar_modelo(numero_classes=len(CLASSES), pretrained=False):
    """Cria a EfficientNet-B0 usada no treinamento e na inferência."""
    pesos = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    modelo = efficientnet_b0(weights=pesos)
    entrada = modelo.classifier[1].in_features
    modelo.classifier[1] = torch.nn.Linear(entrada, numero_classes)
    return modelo


def transformar_avaliacao():
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def recortar_lesao(imagem_bgr, mascara, margem=0.15):
    """
    Recorta a menor região que contém a máscara e preserva uma margem de pele.

    A margem é útil para características próximas da borda, como calosidade.
    """
    pontos = cv2.findNonZero((mascara > 0).astype(np.uint8))
    if pontos is None:
        raise ValueError("A máscara está vazia; não é possível recortar a lesão.")

    x, y, largura, altura = cv2.boundingRect(pontos)
    expansao = int(max(largura, altura) * margem)
    x1 = max(0, x - expansao)
    y1 = max(0, y - expansao)
    x2 = min(imagem_bgr.shape[1], x + largura + expansao)
    y2 = min(imagem_bgr.shape[0], y + altura + expansao)
    return imagem_bgr[y1:y2, x1:x2].copy()


class ClassificadorCaracteristicas:
    """Carrega uma versão aprovada do classificador uma única vez."""

    def __init__(self, checkpoint, device=None):
        self.checkpoint_path = Path(checkpoint)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint do classificador não encontrado: {checkpoint}"
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        pacote = torch.load(self.checkpoint_path, map_location=self.device)
        self.classes = pacote.get("classes", CLASSES)
        if self.classes != CLASSES:
            raise ValueError(
                f"Classes incompatíveis. Esperado {CLASSES}; recebido "
                f"{self.classes}."
            )

        self.thresholds = pacote.get(
            "thresholds", {nome: 0.5 for nome in self.classes}
        )
        self.versao = pacote.get("version", self.checkpoint_path.stem)
        self.modelo = criar_modelo(len(self.classes), pretrained=False)
        self.modelo.load_state_dict(pacote["model_state_dict"])
        self.modelo.to(self.device).eval()
        self.transform = transformar_avaliacao()

    @torch.inference_mode()
    def prever(self, recorte_bgr):
        if recorte_bgr is None or recorte_bgr.size == 0:
            raise ValueError("O recorte recebido pelo classificador está vazio.")

        rgb = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB)
        entrada = self.transform(Image.fromarray(rgb)).unsqueeze(0)
        probabilidades = torch.sigmoid(
            self.modelo(entrada.to(self.device))
        )[0].cpu()

        caracteristicas = {}
        for nome, probabilidade in zip(self.classes, probabilidades.tolist()):
            limite = float(self.thresholds.get(nome, 0.5))
            caracteristicas[nome] = {
                "presente": bool(probabilidade >= limite),
                "probabilidade": round(float(probabilidade), 4),
                "threshold": round(limite, 4),
            }

        return {
            "modelo": "efficientnet_b0_multilabel",
            "versao": self.versao,
            "caracteristicas": caracteristicas,
        }
