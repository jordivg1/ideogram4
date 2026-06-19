# Ideogram 4 — arquitectura explicada

Notas propias sobre cómo funciona el modelo por dentro, con los números reales del
`Ideogram4Config` y del autoencoder de este repo. Pensado para alguien que viene de
LLMs/transformers pero no ha tocado difusión.

---

## 1. Visión general

Ideogram 4 es un modelo **texto → imagen** de tipo **flow matching** (no difusión DDPM
clásica). El pipeline de inferencia es:

```
frase corta
   │  (magic prompt: una LLM)
   ▼
caption JSON estructurado
   │  (Qwen3-VL-8B: text encoder)
   ▼
tokens de condición ─────────────┐
                                  ▼
ruido (32,64,64) ──► DiT (×N pasos, flow matching) ──► latente final
                                  │  (VAE decoder)
                                  ▼
                              imagen RGB
```

Tres piezas entrenadas:

| Pieza | Qué es | Params |
|---|---|---|
| **VAE** (autoencoder) | comprime imagen ↔ latente | pequeño |
| **DiT** | el modelo de difusión (la estrella) | **~9.3B** |
| **Text encoder** | Qwen3-VL-8B-Instruct (un VLM entero) | ~8B |

El "magic prompt" es una llamada a otra LLM (hosted o vía OpenRouter), no es parte de
los pesos.

---

## 2. Flow matching (en vez de difusión clásica)

La difusión clásica (DDPM) entrena la red para **predecir el ruido** `ε` y revierte una
cadena estocástica de cientos de pasos. Ideogram 4 usa **flow matching / rectified flow**:

- Defines un camino recto entre un latente real `x₀` y ruido `x₁`:
  `x_t = (1−t)·x₀ + t·x₁`.
- La velocidad objetivo a lo largo de ese camino es **constante**: `v = x₁ − x₀`.
- La red aprende a predecir esa velocidad `v_θ(x_t, t, condición)`.
- En inferencia resuelves una **ODE**: integras de `t=1` (ruido) a `t=0` (imagen).
  Cada "paso de denoising" es un paso del solver (tipo Euler): `x ← x − Δt · v`.

Como los caminos son casi rectos, bastan **pocos pasos** (12 en turbo, 48 en calidad),
frente a los 50–1000 de DDPM.

Parámetros de muestreo (presets en `sampler_configs.py`):
- `num_steps`: pasos del solver (12 / 20 / 48).
- `guidance_schedule`: **CFG** variando por paso. CFG corre el modelo con condición y sin
  ella y extrapola: `v = v_uncond + w·(v_cond − v_uncond)`. `w` alto = más obediente.
- `mu` / `std`: el **time-shift** del schedule (a más resolución, más shift).

---

## 3. El VAE: por qué pasamos de 3 a 32 canales

### Qué hace
El VAE (autoencoder variacional) es un **compresor aprendido** entre píxeles y un espacio
latente más pequeño:

- **Encoder**: `imagen (3, H, W)` → `latente (32, H/8, W/8)`
- **Decoder**: `latente (32, H/8, W/8)` → `imagen (3, H, W)`

Config real (`autoencoder.py`): `ch_mult = [1, 2, 4, 4]` ⇒ 3 etapas de *downsample* ⇒
**8× espacial**. `z_channels = 32`.

### Por qué 32 canales y no 3
Es un **trade espacial ↔ canales**. Al bajar la resolución 8× pierdes muchísima
información espacial; para no perder calidad, esa información se "reempaqueta" en más
canales. Ejemplo a 512×512:

| | shape | nº de valores |
|---|---|---|
| Imagen RGB | (3, 512, 512) | 786.432 |
| Latente VAE | (32, 64, 64) | 131.072 |

Resultado: **~6× menos valores** en total, pero —lo importante para el DiT— **64× menos
posiciones espaciales** (512² → 64²). Como el coste de la atención es cuadrático en el
número de posiciones, eso lo cambia todo.

Los 3 canales RGB son una representación "tonta" (rojo/verde/azul por píxel). Los 32
canales del latente son **features aprendidas**: cada canal codifica patrones útiles
(bordes, texturas, color a distintas frecuencias). Más canales = latente de mayor
fidelidad. Referencia: SD1.5 usaba 4 canales, FLUX usa 16, Ideogram 4 usa 32 (latentes
más ricos → mejor detalle y texto).

### Por qué "variacional" (VAE y no AE a secas)
El encoder produce una distribución (media + varianza) con una regularización KL ligera
que empuja el latente hacia algo suave y bien-comportado (cercano a gaussiano). Eso hace
que el espacio latente sea un **objetivo agradable de modelar** para el DiT. Un
autoencoder plano podría tener un latente arbitrariamente "picudo", más difícil de
aprender. En la práctica el peso KL es pequeño; los "VAE" modernos para difusión son casi
AEs con regularización suave.

---

## 4. Patchify: cómo se trocea el latente

El DiT es un transformer y necesita una **secuencia de tokens**, no una rejilla 2D. El
patchify convierte la rejilla latente en tokens, igual que un ViT pero sobre el latente.

Config: `patch_size = 2`, y por eso `in_channels = z_channels(32) × patch²(4) = 128`.

Paso a paso, a 512px → latente `(C=32, H=64, W=64)`:

1. **Trocear en bloques 2×2 no solapados.** La rejilla 64×64 se divide en una rejilla de
   `32×32 = 1024` bloques, cada uno de 2×2 posiciones.
2. **Aplanar cada bloque con todos sus canales.** Un bloque tiene `32 canales × 2 × 2 =
   128` valores → se aplana en **un vector de 128**.
3. Quedan **1024 tokens** de dimensión 128.
4. **Proyección lineal** `128 → 4608`: cada token pasa a la dimensión del modelo.

En notación einops:
```python
# (c, h, w) -> (num_tokens, c*p1*p2)
rearrange(x, "c (h p1) (w p2) -> (h w) (c p1 p2)", p1=2, p2=2)
```

Al final de la red se hace lo inverso (**unpatchify**): de `(1024, 128)` se reconstruye la
rejilla `(32, 64, 64)`.

Número de tokens = `(píxeles / 16)²` (8× del VAE × 2 del patch):

| Resolución | rejilla latente | tokens |
|---|---|---|
| 512² | 32×32 | 1.024 |
| 768² | 48×48 | 2.304 |
| 1024² | 64×64 | 4.096 |
| 2048² | 128×128 | 16.384 |

---

## 5. ¿Por qué un VAE? ¿No se puede hacer la inversa del flow?

Pregunta clave, porque mezcla dos "inversas" distintas:

- **El flow (DiT)** mapea **ruido ↔ latente**, ambos en el *mismo* espacio
  `(32,64,64)`. Es (aproximadamente) invertible porque es una ODE determinista. Pero
  **nunca toca píxeles**.
- **El VAE** mapea **píxeles ↔ latente**. Su "inversa" es el **decoder** (que es una red
  aparte, entrenada para reconstruir; el encode→decode es *lossy*, no identidad exacta).

Es decir, hay **dos pares inversa-ish, en dominios diferentes**:

```
píxeles  ⇄  latente   (VAE encoder / VAE decoder)
ruido    ⇄  latente   (flow forward / flow reverse)
```

"Invertir el flow" te lleva de ruido a **latente**, no a píxeles. Para llegar a píxeles
necesitas sí o sí el **decoder del VAE**. Son operaciones distintas.

### ¿Y por qué no correr el flow directamente sobre píxeles (y olvidarse del VAE)?
Se puede (existe la *pixel-space diffusion*), pero es carísimo a alta resolución:

- A 512px en píxeles, con patch 2, tendrías `512×512/4 = 65.536` tokens, frente a `1.024`
  en latente.
- La atención es O(n²): `65.536²` vs `1.024²` ≈ **~4000× más** cómputo de atención.

Además el VAE descarta detalle de alta frecuencia imperceptible, así el DiT gasta su
capacidad en semántica y composición en vez de en reproducir cada píxel. Por eso el
estándar es **difusión latente** (VAE + DiT).

---

## 6. El DiT por dentro

Config real (`Ideogram4Config`):

| Campo | Valor |
|---|---|
| `emb_dim` (dimensión del modelo) | 4608 |
| `num_layers` | 34 |
| `num_heads` | 18 (head_dim = 256) |
| `intermediate_size` (MLP SwiGLU) | 12288 |
| `adanln_dim` (embedding del timestep) | 512 |
| `in_channels` | 128 |
| `rope_theta` | 5.000.000 |
| `mrope_section` | (24, 20, 20) — RoPE multimodal 3D |
| Total params | ~9.3B |

Params por bloque (×34): atención ≈ 85M (`qkv` 4608→13824, `o` 4608→4608), MLP SwiGLU
≈ 170M (`w1,w3` 4608→12288, `w2` 12288→4608), adaLN ≈ 9.4M → **~264M/bloque**.

### El bloque (lo que lo hace un *Diffusion* Transformer: adaLN)
A diferencia de un transformer normal, el timestep `t` **no** entra como token. Una Linear
`512 → 4×4608` genera 4 señales de modulación que escalan las normalizaciones y abren/
cierran los residuales:

```
x ──► RMSNorm ──► ×(1+scale_msa) ──► Self-Attention ──► ×gate_msa ──► (+) ──► x'
x' ─► RMSNorm ──► ×(1+scale_mlp) ──► MLP SwiGLU      ──► ×gate_mlp ──► (+) ──► x''
        ▲                                                    ▲
        └──────────── modulación desde t (adaLN) ───────────┘
```

Así el mismo peso "sabe" si está en el paso 1 (mucho ruido) o en el 11 (casi imagen).

### Single-stream (vs MMDiT de SD3/FLUX)
Texto e imagen se concatenan en **una sola secuencia** procesada por **los mismos pesos**
en las 34 capas, sin ramas separadas. Más interacción cross-modal y escala como un LLM.

### Recorrido de shapes (a 512px)

| Etapa | Shape |
|---|---|
| Ruido latente | (32, 64, 64) |
| Patchify 2×2 | (1024, 128) |
| Proyección | (1024, 4608) |
| + tokens de condición | (L, 4608) |
| qkv (por bloque) | (L, 13824) → 18×(L,256) |
| atención (por cabeza) | (L, L) |
| MLP SwiGLU | (L,4608)→(L,12288)→(L,4608) |
| Cabeza de salida (velocidad) | (L, 128) |
| Unpatchify | (32, 64, 64) |
| (×12 pasos) → VAE decoder | (3, 512, 512) |

---

## 7. El text encoder: un VLM entero

En vez de CLIP/T5, usa **Qwen3-VL-8B-Instruct**. Se extraen hidden states de **13 capas**
(0, 3, 6, …, 33, 35 → `llm_features_dim = 4096 × 13 = 53248`), que se proyectan y entran
al DiT como tokens de condición. Esto da comprensión profunda del prompt, multilingüe, y
sobre todo **renderizado de texto** correcto dentro de la imagen.

---

## 8. Magic prompt

El modelo se entrenó con **captions JSON estructurados**, no con frases planas. El "magic
prompt" es una LLM que expande tu frase a ese JSON. Configs disponibles: `ideogram-4-v1`
(API hosted gratis de Ideogram, lee `IDEOGRAM_API_KEY`), `claude-opus-v1`,
`claude-sonnet-v1` (vía OpenRouter).

Esquema del JSON:
```jsonc
{
  "high_level_description": "...resumen de la escena...",
  "compositional_deconstruction": {
    "background": "...descripción del fondo...",
    "elements": [
      { "type": "obj",  "desc": "...objeto..." },
      { "type": "text", "text": "TEXTO EXACTO", "desc": "...cómo/dónde se renderiza..." }
    ]
  }
}
```

Clave: los elementos `type: "text"` llevan **la cadena literal** a renderizar → por eso
Ideogram escribe texto tan bien. Contrapartida: la LLM **inventa composición** (objetos,
fondo) que tú no pediste. Con `--no-magic-prompt` el modelo recibe tu frase plana (fuera
de su distribución → peor); o puedes escribir tú el JSON a mano para control total.

---

## 9. Notas de ejecución en Mac (Apple Silicon)

Footprint real de los pesos (fp8), medido en caché:

| Componente | fp8 |
|---|---|
| transformer (condicional) | 8.7 GB |
| unconditional_transformer | 8.7 GB |
| text_encoder (Qwen3-VL) | 8.2 GB |
| vae | 0.16 GB |
| **total** | **~26 GB** |

Son **dos DiTs completos** (CFG asimétrico, ver §10.6) + el encoder → **26 GB > 24 GB**
de RAM. En una M-Pro de 24 GB esto **no cabe** ni en fp8 (hace *thrashing* de swap).

Limitaciones de plataforma:
- **fp8 NO corre en MPS**: PyTorch no soporta el dtype `float8_e4m3fn` en Metal
  (`Fp8Linear.forward` revienta en `weight.to(bf16)`).
- La ruta `nf4` (bitsandbytes 4-bit) es **CUDA-only**.

Solución implementada en [`run_metal.py`](../run_metal.py): ver §11.

Licencia **Non-Commercial**: solo uso personal/experimental.

---

## 10. Las matemáticas, en detalle

Notación: `d = 4608` (emb_dim), `d_ff = 12288`, `h = 18` cabezas, `d_h = 256`
(head_dim), `L` = longitud de secuencia, `c` = condición (tokens del encoder).

### 10.1 Flow matching (rectified flow)

Dos extremos: un latente real `x₀ ~ p_data` y ruido `x₁ ~ N(0, I)`. Se define el
**camino recto**:

$$x_t = (1-t)\,x_0 + t\,x_1, \qquad t \in [0,1]$$

Su derivada temporal (la **velocidad**) es constante a lo largo del camino:

$$\frac{dx_t}{dt} = x_1 - x_0 =: u_t$$

La red `v_θ` se entrena para regresar esa velocidad (condicionada en `t` y en `c`):

$$\mathcal{L} = \mathbb{E}_{t,\,x_0,\,x_1}\Big[\;\big\lVert v_\theta(x_t, t, c) - (x_1 - x_0) \big\rVert_2^2 \;\Big]$$

En **inferencia** se resuelve la ODE de `t=1` (ruido) a `t=0` (imagen):

$$\frac{dx}{dt} = v_\theta(x, t, c)$$

Con un solver de Euler y paso `Δt` (lo que hace el bucle, `z = z + v·Δt` con `Δt<0`):

$$x_{t-\Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t, c)$$

Como el camino es recto, la trayectoria es casi una línea → bastan pocos pasos (12–48).

### 10.2 Schedule de `t` (logit-normal + shift por resolución)

Durante el entrenamiento `t` no se muestrea uniforme sino **logit-normal** (params
`μ = mu`, `σ = std`):

$$t = \sigma\!\left(\mu + \sigma\,\varepsilon\right), \quad \varepsilon \sim N(0,1), \quad \sigma(a)=\tfrac{1}{1+e^{-a}}$$

En inferencia se aplica un **time-shift** `s` que depende de la resolución (más tokens →
más shift), que reescala `t`:

$$t' = \frac{s\,t}{1 + (s-1)\,t}$$

Esto concentra pasos donde más hacen falta a alta resolución.

### 10.3 VAE (ELBO)

Encoder probabilístico `q_φ(z|x) = N(μ_φ(x), σ_φ²(x))`, con reparametrización
`z = μ_φ(x) + σ_φ(x) ⊙ ε`, `ε ~ N(0,I)`. Decoder `p_θ(x|z)`. Se maximiza el ELBO:

$$\log p(x) \;\ge\; \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(x|z)]}_{\text{reconstrucción}} \;-\; \underbrace{\mathrm{KL}\!\big(q_\phi(z|x)\,\Vert\,N(0,I)\big)}_{\text{regularización}}$$

$$\mathrm{KL} = \tfrac{1}{2}\sum_i \big(\mu_i^2 + \sigma_i^2 - \log\sigma_i^2 - 1\big)$$

En la práctica la reconstrucción es `L1/L2 + perceptual + GAN`, y el peso de la KL es
pequeño (β≪1) → "VAE" casi-determinista. Antes de difundir, el latente se normaliza
(buffers `latent_shift`, `latent_scale`): `ẑ = (z − shift)/scale`; al decodificar se
invierte: `z = ẑ·scale + shift`.

### 10.4 Patchify (conteo)

Latente `(C, H_l, W_l)` con `C=32`, `H_l=W_l=H/8`. Con `patch=p=2`:

$$\text{tokens} = \frac{H_l}{p}\cdot\frac{W_l}{p}, \qquad \dim_\text{token} = C\cdot p^2 = 32\cdot4 = 128$$

$$\text{rearrange: } (C, H_l, W_l) \;\to\; \Big(\tfrac{H_l}{p}\tfrac{W_l}{p},\; C p^2\Big) \;\xrightarrow{\text{Linear } 128\to d}\; (L_\text{img}, d)$$

### 10.5 Atención + RoPE 3D

Por cabeza, atención escalada:

$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_h}}\Big)\,V, \qquad d_h = 256$$

`qkv` proyecta `d → 3d` y se reparte en `h=18` cabezas de `d_h=256` (`h·d_h = d`).

**RoPE** rota pares de dimensiones por un ángulo dependiente de la posición:

$$\theta_k = \text{pos}\cdot \text{base}^{-2k/d_h}, \qquad \text{base} = 5\times10^{6}$$

**mRoPE 3D**: el `head_dim/2 = 128` se parte en secciones `(24, 20, 20)` para las
coordenadas `(temporal, alto, ancho)`. Tokens de imagen usan `pos = (0, h, w)`; tokens
de texto usan `pos = (p, p, p)`. Así la posición 2D del patch entra en la atención.

Coste: tiempo `O(L²·d)`, memoria de la matriz de scores `O(h·L²)` → cuadrático en `L`
(ver §10.8).

### 10.6 adaLN y CFG

**adaLN.** El embedding del timestep `t` (dim `adanln_dim=512`) pasa por una
`Linear(512 → 4d)` que produce 4 señales: `scale_msa, gate_msa, scale_mlp, gate_mlp`.
Cada sub-capa se modula así (RMSNorm tipo *sandwich*, con `gate = tanh(·)` y
`scale ← 1+scale`, según el código):

$$x' = x + \mathrm{gate_{msa}} \odot \mathrm{Norm_2}\big(\mathrm{Attn}(\mathrm{Norm_1}(x)\odot(1+\mathrm{scale_{msa}}))\big)$$

$$x'' = x' + \mathrm{gate_{mlp}} \odot \mathrm{Norm_2}\big(\mathrm{FF}(\mathrm{Norm_1}(x')\odot(1+\mathrm{scale_{mlp}}))\big)$$

Con `RMSNorm(x) = \dfrac{x}{\sqrt{\overline{x^2}+\epsilon}}\odot\gamma`.

**CFG asimétrico.** Se combinan la velocidad condicional y la incondicional. El código
usa la forma convexa (con `gw` = peso por paso del `guidance_schedule`):

$$v = \mathrm{gw}\cdot v_\text{cond} + (1-\mathrm{gw})\cdot v_\text{uncond} = v_\text{uncond} + \mathrm{gw}\,(v_\text{cond} - v_\text{uncond})$$

"Asimétrico" = la rama incondicional es **solo imagen** (sin tokens de texto) y usa un
**segundo DiT** (`unconditional_transformer`) con la condición a cero. Por eso hay dos
modelos en memoria.

### 10.7 SwiGLU

$$\mathrm{FF}(x) = W_2\big(\mathrm{SiLU}(W_1 x) \odot (W_3 x)\big), \qquad \mathrm{SiLU}(a)=a\,\sigma(a)$$

Con `W₁, W₃ : d→d_ff` y `W₂ : d_ff→d` (`d=4608`, `d_ff=12288`).

### 10.8 Conteo de parámetros y FLOPs

**Params por bloque** (`d=4608`, `d_ff=12288`, `d_a=512`):

$$\underbrace{3d^2 + d^2}_{\text{attn }=4d^2} + \underbrace{3\,d\,d_{ff}}_{\text{MLP}} + \underbrace{d_a\cdot 4d}_{\text{adaLN}} = 84.9\text{M} + 169.9\text{M} + 9.4\text{M} \approx 264\text{M}$$

$$\times\,34 \text{ bloques} \approx 8.99\text{B} \;\;(+\text{ embeds/cabeza}) \approx 9.3\text{B}$$

**FLOPs por forward** (dominado por los Linear): `≈ 2·N_params·L`. A 512px (`L≈1024`):

$$\approx 2 \cdot 9.3\times10^{9} \cdot 1024 \approx 1.9\times10^{13}\ \text{FLOP/forward}$$

La parte de atención `≈ 4 L^2 d` por capa: a `L=1024` son `~6.6×10¹¹` en total (las 34
capas) — un orden de magnitud por debajo de los Linear *a esta resolución*. A 2048px
(`L≈16384`) la atención crece `×256` y pasa a dominar.

**Por imagen**: `num_steps × (1 forward si no-CFG, 2 si CFG)`. Turbo (12) sin CFG ≈
`12 × 1.9×10¹³ ≈ 2.3×10¹⁴` FLOP (230 TFLOP); con CFG, el doble.

### 10.9 Cuantización fp8 (e4m3) y la LUT

Formato `e4m3`: 1 bit signo, 4 exponente (bias 7), 3 mantisa → máx normal `448`.
**Escala por fila** (por canal de salida `i`):

$$s_i = \frac{\max_j |W_{ij}|}{448}, \qquad Q_{ij} = \mathrm{round_{fp8}}\!\Big(\frac{W_{ij}}{s_i}\Big), \qquad W_{ij} \approx Q_{ij}\cdot s_i$$

Como cada `Q_{ij}` es **1 byte**, la descuantización es una **LUT** de 256 entradas
`T[b] = \text{fp8\_to\_float}(b)`:

$$W \approx T\big[\text{bytes}(Q)\big] \odot s$$

Esto es justo lo que hace [`run_metal.py`](../run_metal.py) para correr en MPS: guarda
`Q` como `uint8` (MPS soporta `uint8`, no `float8`) y aplica `T` con un *gather*.

---

## 11. Optimización para Metal (24 GB) — `run_metal.py`

El problema es doble: (a) MPS no soporta `float8_e4m3fn`; (b) los pesos suman ~26 GB y
no caben en 24 GB (§9). `run_metal.py` ataca ambos:

**1. Shim fp8 → MPS por LUT.** Se monkeypatchea `Fp8Linear` para guardar el peso como
`uint8` (los bytes crudos del e4m3, que MPS sí almacena) y descuantizar en el forward con
una LUT de 256 entradas (§10.9):
```python
w = lut[self.weight.long()] * self.weight_scale.unsqueeze(1)   # uint8 -> bf16
```
También se parchea `load_fp8_state_dict` para hacer `tensor.view(torch.uint8).to("mps")`
en lugar de mover el `float8` (que falla).

**2. Carga por fases liberando memoria.** Nunca se tienen los 26 GB a la vez:
```
Fase A: cargar encoder (8.2GB) → codificar prompt → LIBERAR encoder
Fase B: cargar DiT (8.7GB)     → bucle de denoising
Fase C: cargar VAE (0.16GB)    → decode
```
Pico ≈ tamaño del componente mayor (~8.7 GB) en vez de 26 GB.

**3. Sin CFG por defecto.** Se salta el segundo DiT (`--cfg` lo reactiva, pero va al
límite de RAM). Sin CFG, `v = v_cond` directamente; con CFG se carga el
`unconditional_transformer` y se aplica §10.6.

Medidas (M5 Pro, 24 GB, 512px, turbo 12, sin CFG):
- Fase A (encoder, carga + encode): ~131 s.
- Per paso de denoising: _(pendiente — se rellena al terminar)_.

Alternativa no implementada para CFG completo en 24 GB: **carga en streaming** de los
shards (mover tensor a tensor a MPS y liberar) para evitar el pico transitorio
`state_dict (CPU) + modelo (MPS)`, manteniendo los 2 DiTs (~17.4 GB) residentes.
