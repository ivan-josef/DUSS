# Classificador supervisionado de características da lesão

Este pacote adiciona um classificador multirrótulo ao pipeline atual. Uma mesma
imagem pode indicar simultaneamente granulação, esfacelo, necrose, calosidade e
exsudato.

## O classificador é treinado continuamente?

Não a cada atendimento. O fluxo recomendado é:

1. O sistema executa a versão aprovada do classificador.
2. A enfermeira registra sua avaliação.
3. O médico confirma ou corrige os rótulos.
4. Os casos confirmados são acumulados.
5. Periodicamente, um novo modelo candidato é treinado em lote.
6. Ele só substitui o modelo em uso depois de avaliação em um conjunto de teste
   separado e aprovação da equipe.

Nunca use a própria previsão do classificador como rótulo de treinamento.

## Arquivos

- `pipeline_ulcera.py`: segmentação e medição já existentes.
- `avaliar_completo.py`: integra uma imagem, formulário, segmentação, medição e
  classificação.
- `classificador_caracteristicas.py`: arquitetura e inferência.
- `exportar_dataset.py`: converte JSONs confirmados pelo médico em CSV.
- `treinar_classificador.py`: treinamento, validação, teste e versionamento.

## 1. Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Altere `MODEL_PATH` em `pipeline_ulcera.py` ou passe
`--modelo-segmentacao`.

## 2. Executar o atendimento antes do primeiro classificador

O segmentador e as medições já podem funcionar enquanto os dados são coletados:

```bash
python avaliar_completo.py \
  test/ferida.jpg \
  exemplo_formulario_enfermeira.json \
  --pixels-por-cm 125
```

O resultado cria também `recortes_classificador/`, que será usado futuramente
no treinamento.

## 3. Estrutura da confirmação médica

Depois da revisão, o registro deve ter:

```json
{
  "status_revisao_medica": "confirmado",
  "confirmacao_medica": {
    "caracteristicas_visuais": {
      "granulacao": "presente",
      "esfacelo": "ausente",
      "necrose": "ausente",
      "calosidade": "presente",
      "exsudato": "presente"
    }
  }
}
```

Também deve manter `patient_id`, `visit_id` e `recorte_classificador`.
Use `nao_avaliado` quando o médico não puder concluir. O exportador deixará esse
rótulo vazio e a função de perda o ignorará.

## 4. Gerar o dataset

```bash
python exportar_dataset.py \
  resultados_ulceras/json \
  dataset/rotulos.csv
```

Somente avaliações confirmadas pelo médico entram no CSV. Os recortes são
copiados para `dataset/rotulos_imagens/`.

## 5. Treinar a primeira versão

```bash
python treinar_classificador.py \
  --csv dataset/rotulos.csv \
  --output modelos/classificador_v1.pth \
  --version 1.0.0 \
  --epochs 30
```

A separação é feita por paciente para que imagens do mesmo paciente não fiquem
simultaneamente em treino e teste. O script salva:

- `classificador_v1.pth`: checkpoint candidato;
- `classificador_v1.json`: métricas e histórico;
- thresholds individuais aprendidos no conjunto de validação.

Não publique o candidato apenas porque o treinamento terminou. Verifique
precisão, recall, F1 e quantidade de exemplos positivos de cada característica.

## 6. Usar uma versão aprovada

```bash
python avaliar_completo.py \
  test/ferida.jpg \
  exemplo_formulario_enfermeira.json \
  --classificador modelos/classificador_v1.pth \
  --pixels-por-cm 125
```

O JSON final mantém três informações separadas:

- `avaliacao_enfermeira`;
- `previsao_classificador`;
- `confirmacao_medica`.

## 7. Retreinamento periódico

Exporte novamente todos os casos confirmados e gere uma nova versão, sem
sobrescrever a anterior:

```bash
python exportar_dataset.py \
  resultados_ulceras/json \
  dataset_v2/rotulos.csv

python treinar_classificador.py \
  --csv dataset_v2/rotulos.csv \
  --resume modelos/classificador_v1.pth \
  --output modelos/classificador_v2_candidato.pth \
  --version 2.0.0
```

Depois da avaliação, o sistema deve apontar para o novo checkpoint aprovado.
Mantenha as versões anteriores para auditoria e possível reversão.

## Limites importantes

- O classificador é apoio à decisão, não diagnóstico autônomo.
- Wagner e prioridade não devem ser rótulos desse modelo visual.
- Febre, odor, dor e profundidade dependem do exame/formulário, não da foto.
- A validação clínica e o protocolo institucional continuam obrigatórios.
