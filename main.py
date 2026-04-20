import gzip
import struct
import urllib.request
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels":  "train-labels-idx1-ubyte.gz",
    "test_images":   "t10k-images-idx3-ubyte.gz",
    "test_labels":   "t10k-labels-idx1-ubyte.gz",
}
MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

DATA_DIR   = Path("mnist_data")
IMAGES_DIR = Path("images")
DEBUG_DIR  = Path("debug_preview")
MODEL_PATH = DATA_DIR / "mnist_nn.npz"

ROTATION_ANGLES = [-30, -20, -15, -10, -5, 0, 5, 10, 15, 20, 30]


def is_valid_gzip(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def download_mnist() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for _, fname in MNIST_FILES.items():
        out = DATA_DIR / fname
        if out.exists() and not is_valid_gzip(out):
            print(f"[WARN] {fname} is corrupted, removing...")
            out.unlink(missing_ok=True)
        if out.exists():
            continue
        ok = False
        for base in MNIST_MIRRORS:
            url = base + fname
            try:
                print(f"Downloading {fname} from {base} ...")
                urllib.request.urlretrieve(url, out)
                if not is_valid_gzip(out):
                    out.unlink(missing_ok=True)
                    continue
                ok = True
                break
            except Exception as e:
                out.unlink(missing_ok=True)
                print(f"  Mirror failed: {e}")
        if not ok:
            raise RuntimeError(f"Cannot download {fname}")
    print("MNIST dataset is ready.")


def read_idx_gz(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        if magic == 2049:
            struct.unpack(">I", f.read(4))[0]
            return np.frombuffer(f.read(), dtype=np.uint8)
        elif magic == 2051:
            n, rows, cols = struct.unpack(">III", f.read(12))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)
        raise ValueError(f"Unknown IDX magic {magic}")


def load_mnist():
    x_tr = read_idx_gz(DATA_DIR / MNIST_FILES["train_images"])
    y_tr = read_idx_gz(DATA_DIR / MNIST_FILES["train_labels"])
    x_te = read_idx_gz(DATA_DIR / MNIST_FILES["test_images"])
    y_te = read_idx_gz(DATA_DIR / MNIST_FILES["test_labels"])
    return x_tr, y_tr, x_te, y_te


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class NeuralNetwork:
    def __init__(self, input_nodes: int, hidden_nodes: int, output_nodes: int, lr: float):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = lr
        self.wih = np.random.normal(0.0, pow(input_nodes,  -0.5), (hidden_nodes, input_nodes))
        self.who = np.random.normal(0.0, pow(hidden_nodes, -0.5), (output_nodes, hidden_nodes))

    def train(self, inputs_list: np.ndarray, targets_list: np.ndarray) -> None:
        inputs     = np.array(inputs_list, ndmin=2).T
        targets    = np.array(targets_list, ndmin=2).T
        hidden_out = sigmoid(self.wih @ inputs)
        final_out  = sigmoid(self.who @ hidden_out)
        out_err    = targets - final_out
        hidden_err = self.who.T @ out_err
        self.who  += self.lr * (out_err    * final_out  * (1.0 - final_out))  @ hidden_out.T
        self.wih  += self.lr * (hidden_err * hidden_out * (1.0 - hidden_out)) @ inputs.T

    def query(self, inputs_list: np.ndarray) -> np.ndarray:
        inputs     = np.array(inputs_list, ndmin=2).T
        hidden_out = sigmoid(self.wih @ inputs)
        return sigmoid(self.who @ hidden_out)

    def save(self, path: Path) -> None:
        np.savez(str(path), wih=self.wih, who=self.who, lr=self.lr,
                 inodes=self.inodes, hnodes=self.hnodes, onodes=self.onodes)

    @staticmethod
    def load(path: Path) -> "NeuralNetwork":
        d  = np.load(str(path), allow_pickle=True)
        nn = NeuralNetwork(int(d["inodes"]), int(d["hnodes"]), int(d["onodes"]), float(d["lr"]))
        nn.wih = d["wih"]
        nn.who = d["who"]
        return nn


def scale_inputs(x: np.ndarray) -> np.ndarray:
    x = x.reshape(x.shape[0], -1).astype(np.float32)
    return (x / 255.0) * 0.99 + 0.01


def one_hot(labels: np.ndarray) -> np.ndarray:
    t = np.full((labels.shape[0], 10), 0.01, dtype=np.float32)
    t[np.arange(labels.shape[0]), labels] = 0.99
    return t


def evaluate(nn: NeuralNetwork, x: np.ndarray, y: np.ndarray) -> float:
    correct = 0
    for i in range(y.shape[0]):
        if int(np.argmax(nn.query(x[i]))) == int(y[i]):
            correct += 1
    return correct / y.shape[0]


def rof_denoise(image: np.ndarray, weight: float = 0.08, n_iter: int = 80) -> np.ndarray:
    """ROF total-variation denoising via Chambolle dual algorithm. image in [0,1]."""
    f  = image.astype(np.float64)
    px = np.zeros_like(f)
    py = np.zeros_like(f)
    tau = 0.25
    for _ in range(n_iter):
        div_p = np.zeros_like(f)
        div_p[1:,  :] += px[1:,  :] - px[:-1, :]
        div_p[0,   :] += px[0,   :]
        div_p[:,  1:] += py[:,  1:] - py[:, :-1]
        div_p[:,   0] += py[:,   0]
        u = f + weight * div_p
        gx = np.zeros_like(f)
        gy = np.zeros_like(f)
        gx[:-1, :] = u[1:, :] - u[:-1, :]
        gy[:, :-1] = u[:, 1:] - u[:, :-1]
        denom = np.maximum(1.0, np.sqrt(gx**2 + gy**2) / weight)
        px = (px + tau * gx) / denom
        py = (py + tau * gy) / denom
    div_p = np.zeros_like(f)
    div_p[1:,  :] += px[1:,  :] - px[:-1, :]
    div_p[0,   :] += px[0,   :]
    div_p[:,  1:] += py[:,  1:] - py[:, :-1]
    div_p[:,   0] += py[:,   0]
    return np.clip(f + weight * div_p, 0.0, 1.0)


def otsu_threshold(arr: np.ndarray) -> float:
    """Otsu's threshold for a float [0,1] grayscale array."""
    hist, edges = np.histogram(arr.ravel(), bins=256, range=(0.0, 1.0))
    hist    = hist.astype(np.float64)
    total   = hist.sum()
    if total == 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0, s0, s_total = 0.0, 0.0, float(np.dot(hist, centers))
    best_var, best_t = -1.0, 0.5
    for i in range(len(hist)):
        w0 += hist[i];  w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        s0   += hist[i] * centers[i]
        mu0   = s0 / w0;  mu1 = (s_total - s0) / w1
        var_b = w0 * w1 * (mu0 - mu1) ** 2
        if var_b > best_var:
            best_var = var_b;  best_t = centers[i]
    return float(best_t)


def photo_to_mnist_vector(gray_pil: Image.Image,
                          save_path: Path = None) -> np.ndarray:
    """
    Convert a grayscale PIL image to a 784-dim MNIST-compatible vector:
      1. Autocontrast  — normalize pixel range
      2. ROF denoise   — remove image noise (Rudin-Osher-Fatemi model)
      3. Otsu threshold — separate digit from background
      4. Bounding box crop
      5. Fit digit into 20x20 inside 28x28 canvas (aspect-ratio preserved)
      6. Center of mass centering
      7. Scale to [0.01, 1.00]
    If save_path is given, saves the resulting 28x28 image as PNG.
    """
    enhanced = ImageOps.autocontrast(gray_pil, cutoff=1)
    arr      = np.array(enhanced).astype(np.float32) / 255.0

    arr = rof_denoise(arr).astype(np.float32)

    bg     = float(np.median(np.concatenate([arr[0], arr[-1],
                                              arr[:, 0], arr[:, -1]])))
    thresh = otsu_threshold(arr)

    if bg > 0.5:
        mask  = arr < thresh
        digit = 1.0 - arr
    else:
        mask  = arr > thresh
        digit = arr.copy()

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return np.full(784, 0.01, dtype=np.float32)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    crop = digit[rmin:rmax + 1, cmin:cmax + 1]
    ch, cw = crop.shape

    scale  = 20.0 / max(ch, cw)
    new_h  = max(1, int(round(ch * scale)))
    new_w  = max(1, int(round(cw * scale)))

    pil_scaled = Image.fromarray((crop * 255).astype(np.uint8)).resize(
        (new_w, new_h), Image.Resampling.LANCZOS)
    scaled = np.array(pil_scaled).astype(np.float32) / 255.0

    canvas = np.zeros((28, 28), dtype=np.float32)
    r_off  = (28 - new_h) // 2
    c_off  = (28 - new_w) // 2
    canvas[r_off:r_off + new_h, c_off:c_off + new_w] = scaled

    binary = (canvas > 0.25).astype(np.float32)
    total  = binary.sum()
    if total > 0:
        cy = float(np.dot(np.arange(28), binary.sum(axis=1)) / total)
        cx = float(np.dot(np.arange(28), binary.sum(axis=0)) / total)
        dr = int(round(14 - cy))
        dc = int(round(14 - cx))
        if dr != 0:
            canvas = np.roll(canvas, dr, axis=0)
            if dr > 0: canvas[:dr, :] = 0
            else:      canvas[dr:, :] = 0
        if dc != 0:
            canvas = np.roll(canvas, dc, axis=1)
            if dc > 0: canvas[:, :dc] = 0
            else:      canvas[:, dc:] = 0

    canvas = np.clip(canvas, 0.0, 1.0)

    if save_path is not None:
        preview = ((1.0 - canvas) * 255).astype(np.uint8)
        Image.fromarray(preview).save(str(save_path))

    canvas = (canvas * 0.99 + 0.01).astype(np.float32)
    return canvas.reshape(-1)


def predict_with_rotation(nn: NeuralNetwork, image_path: Path,
                          save_debug: bool = False) -> int:
    img      = Image.open(image_path).convert("L")
    arr_orig = np.array(img).astype(np.float32) / 255.0
    bg       = float(np.median(np.concatenate([arr_orig[0], arr_orig[-1],
                                                arr_orig[:, 0], arr_orig[:, -1]])))
    bg_fill  = int(bg * 255)

    if save_debug:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    best_digit = -1
    best_conf  = -1.0
    results    = []

    for angle in ROTATION_ANGLES:
        rotated    = img.rotate(angle, expand=True, fillcolor=bg_fill)
        debug_path = (DEBUG_DIR / f"{image_path.stem}_angle{angle:+d}.png"
                      if save_debug else None)
        vec        = photo_to_mnist_vector(rotated, save_path=debug_path)
        scores     = nn.query(vec).reshape(-1)
        digit      = int(np.argmax(scores))
        conf       = float(scores[digit])
        results.append((angle, digit, conf))
        if conf > best_conf:
            best_conf  = conf
            best_digit = digit

    print(f"\nImage: {image_path.name}")
    print(f"{'Angle':>7}  {'Digit':>5}  {'Confidence':>10}")
    print("-" * 28)
    for angle, digit, conf in results:
        marker = " <--" if (digit == best_digit and conf == best_conf) else ""
        print(f"{angle:>7}  {digit:>5}  {conf:>10.4f}{marker}")

    print(f"\n>>> Best prediction: digit = {best_digit}  "
          f"(confidence = {best_conf:.4f})")

    if save_debug:
        print(f"    28x28 previews saved to ./{DEBUG_DIR}/")

    return best_digit


def get_image_files() -> list:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )


def choose_image(files: list) -> Path:
    print(f"Images found in ./{IMAGES_DIR}/:")
    for i, p in enumerate(files, 1):
        print(f"  {i}. {p.name}")
    try:
        choice = input("\nEnter image name (or number): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            return files[idx]
        print("[ERROR] Invalid number.")
        return None
    name = choice
    if not Path(name).suffix:
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            if (IMAGES_DIR / (name + ext)).exists():
                name += ext;  break
    p = IMAGES_DIR / name
    if not p.exists():
        print(f"[ERROR] File not found: {p}")
        return None
    return p


def menu_train():
    print("\n=== TRAINING ===")
    try:
        epochs_in = input("Epochs       [default 5]:   ").strip()
        hidden_in = input("Hidden nodes [default 200]: ").strip()
        lr_in     = input("Learning rate [default 0.1]: ").strip()
        seed_in   = input("Random seed  [default 42]:  ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.");  return

    epochs = int(epochs_in) if epochs_in.isdigit() else 5
    hidden = int(hidden_in) if hidden_in.isdigit() else 200
    lr     = float(lr_in)   if lr_in               else 0.1
    seed   = int(seed_in)   if seed_in.isdigit()   else 42

    np.random.seed(seed)
    download_mnist()

    x_tr_img, y_tr, x_te_img, y_te = load_mnist()
    x_tr = scale_inputs(x_tr_img)
    x_te = scale_inputs(x_te_img)
    t_tr = one_hot(y_tr)

    nn = NeuralNetwork(784, hidden, 10, lr)
    print(f"\nTraining: epochs={epochs}, hidden={hidden}, lr={lr}")
    for e in range(1, epochs + 1):
        idx = np.random.permutation(x_tr.shape[0])
        for i in idx:
            nn.train(x_tr[i], t_tr[i])
        acc = evaluate(nn, x_te, y_te)
        print(f"  Epoch {e}/{epochs} -> test accuracy: {acc:.4f}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    nn.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


def menu_predict(save_debug: bool = False):
    print("\n=== PREDICT ===")
    if not MODEL_PATH.exists():
        print("[ERROR] Model not found. Train first (option 1).");  return
    files = get_image_files()
    if not files:
        print(f"[INFO] No images in ./{IMAGES_DIR}/");  return
    image_path = choose_image(files)
    if image_path is None:
        return
    print(f"\nLoading model from {MODEL_PATH} ...")
    nn = NeuralNetwork.load(MODEL_PATH)
    predict_with_rotation(nn, image_path, save_debug=save_debug)


def main():
    print("=" * 45)
    print("  MNIST Digit Recognition — Lab 2")
    print("  ROF denoising + multi-angle prediction")
    print("=" * 45)

    while True:
        print("\nMenu:")
        print("  1 — Train neural network")
        print("  2 — Recognize image from ./images/")
        print("  3 — Recognize image + save 28x28 previews to ./debug_preview/")
        print("  0 — Exit")

        try:
            choice = input("\nYour choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.");  break

        if   choice == "0": print("Goodbye!"); break
        elif choice == "1": menu_train()
        elif choice == "2": menu_predict(save_debug=False)
        elif choice == "3": menu_predict(save_debug=True)
        else: print("[WARN] Unknown option. Please enter 0, 1, 2, or 3.")


if __name__ == "__main__":
    main()