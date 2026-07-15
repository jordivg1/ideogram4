# Diagrams

This page collects the model and inference diagrams that are useful when
orienting around the Ideogram 4 codebase and comparing it with Krea 2.

## Repo Map

```mermaid
flowchart TD
  R["ideogram4 repo"] --> Run["run_inference.py / run_metal.py"]
  R --> Src["src/ideogram4"]
  R --> Docs["docs"]
  R --> Assets["assets"]

  Src --> Pipe["pipeline_ideogram4.py"]
  Src --> Model["modeling_ideogram4.py"]
  Src --> Sched["scheduler.py"]
  Src --> AE["autoencoder.py"]
  Src --> Safety["safety.py"]
  Src --> Magic["magic_prompt.py"]

  Pipe --> Model
  Pipe --> Sched
  Pipe --> AE
  Pipe --> Magic
  Pipe --> Safety
```

## Ideogram 4 Inference

```mermaid
sequenceDiagram
  participant User
  participant Magic as Magic Prompt
  participant Qwen as Qwen3-VL
  participant DiT as Ideogram4Transformer
  participant Sampler as Euler Sampler
  participant VAE

  User->>Magic: plain prompt
  Magic->>Qwen: structured JSON caption
  Qwen->>DiT: concatenated hidden-state taps
  Sampler->>DiT: noisy image tokens + timestep
  DiT->>Sampler: conditional velocity
  Sampler->>DiT: image-only negative branch
  DiT->>Sampler: unconditional velocity
  Sampler->>Sampler: asymmetric CFG update
  Sampler->>VAE: final latents
  VAE->>User: image
```

## Transformer Block

```mermaid
flowchart TD
  X["Token sequence"] --> N1["RMSNorm"]
  T["Timestep embedding"] --> M["AdaLN projection"]
  M --> SA["Scale attention branch"]
  N1 --> SA
  SA --> A["Attention: QK-RMSNorm + MRoPE"]
  A --> G1["Tanh gate + residual"]
  X --> G1
  G1 --> N2["RMSNorm"]
  M --> SM["Scale MLP branch"]
  N2 --> SM
  SM --> FF["SwiGLU MLP"]
  FF --> G2["Tanh gate + residual"]
  G1 --> G2
```

## Asymmetric CFG

```mermaid
flowchart LR
  Z["Current latents"] --> Pos["Conditional transformer: text + image tokens"]
  Z --> Neg["Unconditional transformer: image tokens only"]
  Text["Qwen3-VL features"] --> Pos
  Zero["Zero text features"] --> Neg
  Pos --> PV["positive velocity"]
  Neg --> NV["negative velocity"]
  PV --> Mix["v = guidance * pos + (1 - guidance) * neg"]
  NV --> Mix
  Mix --> Step["Euler flow step"]
```

## Ideogram 4 vs Krea 2

```mermaid
flowchart TD
  Root["Modern image model stack"]

  Root --> I["Ideogram 4"]
  I --> ICode["Open inference code"]
  I --> IParams["9.3B released variants"]
  I --> IJson["JSON captions"]
  I --> IBox["Bounding boxes"]
  I --> IColor["Color palettes"]
  I --> ICFG["Asymmetric CFG"]
  I --> I2K["Native 2K focus"]

  Root --> K["Krea 2"]
  K --> KReport["Technical report focus"]
  K --> KData["Data curation"]
  K --> KMid["Midtraining and SFT"]
  K --> KPref["Preference optimization"]
  K --> KRL["Multi-reward RL"]
  K --> KPrompt["Prompt expander"]
  K --> KStyle["Style references"]
```

## Krea 2 Training Stack

```mermaid
flowchart TD
  Raw["Large real-image corpus"] --> Filter["Dedup, filtering, OCR, captioning"]
  Filter --> Pre["Progressive pretraining: 256, 512, 1024"]
  Pre --> Mid["Midtraining for domain coverage"]
  Mid --> SFT["High-aesthetic SFT"]
  SFT --> Merge["Model merging"]
  Merge --> PO["Preference optimization"]
  PO --> RL["Multi-reward RL"]
  RL --> Distill["Optional timestep distillation"]
  Distill --> Serve["Generation stack"]

  Expand["Prompt expander"] --> Serve
  Style["Style-reference system"] --> Serve
```
