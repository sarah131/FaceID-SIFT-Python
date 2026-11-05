import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from collections import defaultdict

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class SIFTRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceID | SIFT Recognition Studio")
        self.root.geometry("1040x760")
        self.root.minsize(900, 650)
        self.root.configure(bg="#F3F6FB")

        self.sift = cv2.SIFT_create(nfeatures=1800)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        # OpenCV ships these Haar cascade XML files with the Python package.
        # Load Haar cascades only if their XML files actually exist.
        # Some OpenCV wheel versions do not include the cascade XML data.
        cascade_dir = Path(cv2.data.haarcascades)
        frontal_xml = cascade_dir / "haarcascade_frontalface_default.xml"
        profile_xml = cascade_dir / "haarcascade_profileface.xml"

        self.face_cascade = cv2.CascadeClassifier(
            str(frontal_xml) if frontal_xml.is_file() else ""
        )
        self.profile_cascade = cv2.CascadeClassifier(
            str(profile_xml) if profile_xml.is_file() else ""
        )

        self.has_frontal_cascade = (
            frontal_xml.is_file() and not self.face_cascade.empty()
        )
        self.has_profile_cascade = (
            profile_xml.is_file() and not self.profile_cascade.empty()
        )

        self.train_folder = tk.StringVar(
            value=str((Path(__file__).parent / "datasets" / "train").resolve())
        )
        self.test_folder = tk.StringVar(
            value=str((Path(__file__).parent / "datasets" / "test").resolve())
        )
        self.status = tk.StringVar(value="Ready — load the training dataset to begin.")
        self.samples = []
        self._configure_style()
        self._build_ui()

        if not self.has_frontal_cascade and not self.has_profile_cascade:
            self._log(
                "WARNING: Haar cascade XML files are missing. "
                "Face detection is unavailable; the app will use a central-crop fallback. "
                "Install the recommended OpenCV version and restart for face-only SIFT."
            )
        elif not self.has_frontal_cascade:
            self._log("NOTICE: Frontal-face cascade is unavailable; profile detection only.")
        elif not self.has_profile_cascade:
            self._log("NOTICE: Profile-face cascade is unavailable; frontal detection only.")

    def _configure_style(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "TFrame", background="#F3F6FB"
        )
        style.configure(
            "Card.TFrame", background="#FFFFFF", relief="flat"
        )
        style.configure(
            "TLabel", background="#F3F6FB", foreground="#25324A",
            font=("Segoe UI", 10)
        )
        style.configure(
            "Card.TLabel", background="#FFFFFF", foreground="#25324A",
            font=("Segoe UI", 10)
        )
        style.configure(
            "Title.TLabel", background="#172554", foreground="#FFFFFF",
            font=("Segoe UI", 22, "bold")
        )
        style.configure(
            "Subtitle.TLabel", background="#172554", foreground="#DCE7FF",
            font=("Segoe UI", 10)
        )
        style.configure(
            "Section.TLabel", background="#FFFFFF", foreground="#172554",
            font=("Segoe UI", 12, "bold")
        )
        style.configure(
            "Primary.TButton", font=("Segoe UI", 10, "bold"),
            padding=(14, 10), background="#2563EB", foreground="#FFFFFF"
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#1D4ED8"), ("disabled", "#A5B4FC")],
            foreground=[("disabled", "#FFFFFF")]
        )
        style.configure(
            "Secondary.TButton", font=("Segoe UI", 10),
            padding=(12, 9), background="#E8EEF9", foreground="#1E3A8A"
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#D7E3F8")]
        )
        style.configure(
            "TEntry", padding=8, fieldbackground="#F8FAFC",
            foreground="#25324A"
        )

    def _build_ui(self):
        page = ttk.Frame(self.root, padding=18)
        page.pack(fill="both", expand=True)

        # Header banner
        header = tk.Frame(page, bg="#172554", padx=22, pady=18)
        header.pack(fill="x", pady=(0, 16))
        tk.Label(
            header, text="FaceID  |  SIFT Recognition Studio",
            bg="#172554", fg="#FFFFFF",
            font=("Segoe UI", 22, "bold")
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Feature-based face identification  •  Euclidean (L2) matching  •  10-person dataset",
            bg="#172554", fg="#DCE7FF",
            font=("Segoe UI", 10)
        ).pack(anchor="w", pady=(5, 0))

        # Dataset paths card
        paths = ttk.Frame(page, style="Card.TFrame", padding=16)
        paths.pack(fill="x", pady=(0, 12))
        ttk.Label(paths, text="DATASET LOCATIONS", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )
        self._path_row(paths, 1, "Training folder", self.train_folder, self.pick_train)
        self._path_row(paths, 2, "Testing folder", self.test_folder, self.pick_test)
        paths.columnconfigure(1, weight=1)

        # Action buttons
        actions = ttk.Frame(page, style="TFrame")
        actions.pack(fill="x", pady=(0, 12))
        ttk.Button(
            actions, text="1  Load Training Features",
            style="Primary.TButton", command=self.load_training
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            actions, text="2  Recognize an Image",
            style="Secondary.TButton", command=self.single
        ).pack(side="left", padx=8)
        ttk.Button(
            actions, text="3  Evaluate Test Set",
            style="Secondary.TButton", command=self.evaluate
        ).pack(side="left", padx=8)

        # Result card
        result_card = ttk.Frame(page, style="Card.TFrame", padding=16)
        result_card.pack(fill="x", pady=(0, 12))
        ttk.Label(result_card, text="RECOGNITION RESULT", style="Section.TLabel").pack(anchor="w")
        self.result = tk.Label(
            result_card, text="Your prediction will appear here",
            bg="#FFFFFF", fg="#1D4ED8",
            font=("Segoe UI", 17, "bold"), anchor="w"
        )
        self.result.pack(fill="x", pady=(10, 3))
        self.details = tk.Label(
            result_card, text="No image has been evaluated yet.",
            bg="#FFFFFF", fg="#64748B",
            font=("Segoe UI", 10), anchor="w"
        )
        self.details.pack(fill="x")

        # Log card
        log_card = ttk.Frame(page, style="Card.TFrame", padding=12)
        log_card.pack(fill="both", expand=True)
        ttk.Label(log_card, text="ACTIVITY & EVALUATION LOG", style="Section.TLabel").pack(
            anchor="w", pady=(0, 8)
        )
        log_frame = tk.Frame(log_card, bg="#FFFFFF")
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(
            log_frame, height=14, wrap="word",
            font=("Consolas", 9), bg="#F8FAFC", fg="#334155",
            insertbackground="#1D4ED8", relief="flat",
            padx=12, pady=10
        )
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scrollbar.set)
        self.log.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Footer/status strip
        footer = tk.Frame(page, bg="#E8EEF9", padx=12, pady=8)
        footer.pack(fill="x", pady=(12, 0))
        tk.Label(
            footer, textvariable=self.status, bg="#E8EEF9",
            fg="#1E3A8A", font=("Segoe UI", 9)
        ).pack(anchor="w")

    def _path_row(self, parent, row, label, var, command):
        ttk.Label(parent, text=label, style="Card.TLabel").grid(
            row=row, column=0, sticky="w", padx=(0, 12), pady=6
        )
        ttk.Entry(parent, textvariable=var).grid(
            row=row, column=1, sticky="ew", padx=4, pady=6
        )
        ttk.Button(
            parent, text="Browse", style="Secondary.TButton",
            command=command
        ).grid(row=row, column=2, padx=(8, 0), pady=6)

    def _log(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")

    def pick_train(self):
        folder = filedialog.askdirectory(title="Select training dataset folder")
        if folder:
            self.train_folder.set(folder)

    def pick_test(self):
        folder = filedialog.askdirectory(title="Select testing dataset folder")
        if folder:
            self.test_folder.set(folder)

    def files(self, root):
        folder = Path(root)
        if not folder.exists():
            return []
        return sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTS
        )

    def descriptors(self, path):
        """Detect a face, crop it, normalize size, and extract SIFT descriptors."""
        image = cv2.imread(str(path))
        if image is None:
            return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Reduce very large images while preserving their aspect ratio.
        scale = min(1.0, 1200.0 / max(h, w))
        if scale < 1.0:
            gray = cv2.resize(
                gray, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        # Detect frontal and profile faces.
        candidates = []

        # Never call detectMultiScale on an empty/unloaded classifier.
        if self.has_frontal_cascade:
            frontal = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            candidates.extend(list(frontal))

        if self.has_profile_cascade:
            profiles = self.profile_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
            )
            candidates.extend(list(profiles))

            # Profile cascade is usually directional; check a flipped image too.
            flipped = cv2.flip(gray, 1)
            reverse_profiles = self.profile_cascade.detectMultiScale(
                flipped, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
            )
            for x, y, fw, fh in reverse_profiles:
                candidates.append((gray.shape[1] - x - fw, y, fw, fh))

        if not candidates:
            # Fallback: use the central region so the app can still process an image.
            # This is less reliable; the log clearly reports the fallback.
            self._log(f"WARNING: Haar face not detected in {Path(path).name}; using central crop.")
            side = int(min(gray.shape) * 0.72)
            cx, cy = gray.shape[1] // 2, gray.shape[0] // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(gray.shape[1], x1 + side)
            y2 = min(gray.shape[0], y1 + side)
            face = gray[y1:y2, x1:x2]
        else:
            # Use the largest detected face.
            x, y, fw, fh = max(candidates, key=lambda box: box[2] * box[3])

            # Expand slightly to retain forehead, cheeks and chin.
            cx, cy = x + fw // 2, y + fh // 2
            side = int(max(fw, fh) * 1.45)
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(gray.shape[1], x1 + side)
            y2 = min(gray.shape[0], y1 + side)
            face = gray[y1:y2, x1:x2]

        if face.size == 0:
            return None

        face = cv2.resize(face, (256, 256), interpolation=cv2.INTER_AREA)
        _, descriptors = self.sift.detectAndCompute(face, None)
        return descriptors

    def load_training(self):
        self.samples = []
        counts = defaultdict(int)
        files = self.files(self.train_folder.get())

        self.log.delete("1.0", "end")
        self._log("TRAINING DATASET")
        self._log("----------------")

        for image_path in files:
            label = image_path.parent.name
            desc = self.descriptors(image_path)
            if desc is not None and len(desc) >= 2:
                self.samples.append((label, image_path, desc))
                counts[label] += 1

        self._log(f"Training images found: {len(files)}")
        self._log(f"Usable face/SIFT images: {len(self.samples)}")
        for label in sorted(counts):
            self._log(f"{label}: {counts[label]} usable images")

        if not self.samples:
            messagebox.showerror(
                "Training error",
                "No usable SIFT descriptors were found. Check the training folder and face detection."
            )
            return

        self.status.set(f"Training features loaded from {len(self.samples)} images.")
        self.result.config(text="Training features loaded successfully.", fg="#15803D")
        self.details.config(text="You can now recognize an image or evaluate the test dataset.")
        messagebox.showinfo(
            "Training complete",
            f"Loaded SIFT features from {len(self.samples)} images."
        )

    def predict(self, path):
        query_desc = self.descriptors(path)
        if query_desc is None or len(query_desc) < 2:
            return "Unknown", 0

        # Collect the best score from each training image for each identity.
        person_scores = defaultdict(list)

        for label, train_path, train_desc in self.samples:
            if train_desc is None or len(train_desc) < 2:
                continue

            pairs = self.matcher.knnMatch(query_desc, train_desc, k=2)
            good = [
                m for pair in pairs if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.75 * n.distance
            ]
            person_scores[label].append(len(good))

        if not person_scores:
            return "Unknown", 0

        # Aggregate the three strongest image-level scores for each person.
        final_scores = {}
        for person, scores in person_scores.items():
            top_scores = sorted(scores, reverse=True)[:3]
            final_scores[person] = sum(top_scores)

        predicted = max(final_scores, key=final_scores.get)
        best_score = final_scores[predicted]

        # A modest rejection threshold; validate/tune with independent test images.
        if best_score < 8:
            return "Unknown", best_score

        return predicted, best_score

    def single(self):
        if not self.samples:
            messagebox.showwarning("Training required", "Load training features first.")
            return

        filename = filedialog.askopenfilename(
            title="Select an image to recognize",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not filename:
            return

        label, score = self.predict(filename)
        self.result.config(
            text=f"Predicted identity: {label}",
            fg="#15803D" if label != "Unknown" else "#B45309"
        )
        self.details.config(
            text=f"Aggregated good SIFT matches: {score}  |  Image: {Path(filename).name}"
        )
        self.status.set("Single-image recognition completed.")
        self._log(f"Single image: {Path(filename).name} -> {label} (score={score})")

    def evaluate(self):
        if not self.samples:
            messagebox.showwarning("Training required", "Load training features first.")
            return

        test_files = self.files(self.test_folder.get())
        if not test_files:
            messagebox.showwarning(
                "No test images",
                "Add test images under test/P1 ... test/P10."
            )
            return

        correct = 0
        total = 0
        per_person = defaultdict(lambda: [0, 0])

        self._log("\nTEST DATASET EVALUATION")
        self._log("-----------------------")

        for image_path in test_files:
            actual = image_path.parent.name
            predicted, score = self.predict(image_path)
            total += 1
            per_person[actual][1] += 1

            is_correct = predicted == actual
            if is_correct:
                correct += 1
                per_person[actual][0] += 1

            self._log(
                f"{image_path.name}: actual={actual}, predicted={predicted}, "
                f"matches={score}, {'CORRECT' if is_correct else 'INCORRECT'}"
            )

        accuracy = (correct / total * 100.0) if total else 0.0
        self._log("\nPER-PERSON RESULTS")
        for label in sorted(per_person):
            c, n = per_person[label]
            self._log(f"{label}: {c}/{n} correct")

        self._log(f"\nAccuracy = ({correct}/{total}) x 100 = {accuracy:.2f}%")
        self.result.config(
            text=f"Test accuracy: {accuracy:.2f}%",
            fg="#15803D" if accuracy >= 70 else "#B45309"
        )
        self.details.config(text=f"Correct predictions: {correct} / {total}")
        self.status.set(f"Evaluation completed on {total} test images.")

        messagebox.showinfo(
            "Evaluation complete",
            f"Accuracy: {accuracy:.2f}%\nCorrect predictions: {correct}/{total}"
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = SIFTRecognitionApp(root)
    root.mainloop()
