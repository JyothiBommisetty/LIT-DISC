import math
import numpy as np
from src.models import LDRIScore, NarrativeEmbedding
from src.config import LDRIParams, DEFAULT_LDRI_PARAMS


class LDRICalculator:
    def __init__(self, params: LDRIParams | None = None):
        self.params = params or DEFAULT_LDRI_PARAMS

    def compute(self, embedding: NarrativeEmbedding,
                confidences: dict[str, float] | None = None,
                mem_avg_beta: float = 0.0) -> LDRIScore:
        activations = self.compute_activations(embedding)
        R_signal = self.signal_richness(activations)
        C_factor = self.coupling_factor(embedding.H_pairs)
        T_factor = self.resonance_factor(embedding.H_res)
        M_factor = self.memory_factor(mem_avg_beta)
        C_tilde = (1.0 + C_factor) / 2.0
        T_tilde = (1.0 + T_factor) / 2.0
        M_tilde = (1.0 + M_factor) / 2.0
        Gamma = self.global_confidence(confidences or {})
        raw = self._ldri_raw(R_signal, C_tilde, T_tilde, M_tilde)
        ldri_val = Gamma * raw
        reliability = self._estimate_reliability(embedding)
        components, fractions = self._decompose(R_signal, C_tilde, T_tilde,
                                                 M_tilde, Gamma, raw, ldri_val)
        return LDRIScore(
            ldri=float(ldri_val),
            gamma_global=float(Gamma),
            reliability=float(reliability),
            raw=float(raw),
            components=components,
            fractions=fractions,
        )

    def compute_activations(self, embedding: NarrativeEmbedding) -> np.ndarray:
        dim_names = ["S", "N", "I", "Sy", "E", "Em"]
        a_vals = []
        sigma = self.params.sigma_max
        for d in dim_names:
            Hd = getattr(embedding, f"H_{d}", None)
            if Hd is None or len(Hd) == 0:
                a_vals.append(0.0)
            else:
                norm = float(np.linalg.norm(Hd))
                a_d = 1.0 - math.exp(-norm ** 2 / (2.0 * sigma ** 2))
                a_vals.append(a_d)
        return np.array(a_vals, dtype=float)

    def signal_richness(self, activations: np.ndarray) -> float:
        return float(np.linalg.norm(activations) / math.sqrt(6.0))

    def coupling_factor(self, H_pairs: np.ndarray) -> float:
        norm = float(np.linalg.norm(H_pairs))
        max_norm = math.sqrt(len(H_pairs))
        return min(1.0, norm / max(1e-10, max_norm))

    def resonance_factor(self, H_res: np.ndarray) -> float:
        norm = float(np.linalg.norm(H_res))
        max_norm = math.sqrt(len(H_res))
        return min(1.0, norm / max(1e-10, max_norm))

    def memory_factor(self, mem_avg_beta: float) -> float:
        return min(1.0, max(0.0, mem_avg_beta))

    def global_confidence(self, confidences: dict[str, float]) -> float:
        if not confidences:
            return 0.85
        prod = 1.0
        for v in confidences.values():
            prod *= min(1.0, max(0.0, v))
        return float(prod)

    def _ldri_raw(self, R_signal: float, C_tilde: float,
                  T_tilde: float, M_tilde: float) -> float:
        p = self.params
        S = p.alpha + p.beta + p.gamma + p.delta
        product = (R_signal ** p.alpha) * (C_tilde ** p.beta) * (T_tilde ** p.gamma) * (M_tilde ** p.delta)
        if product < 1e-10:
            return 0.0
        return float(product ** (1.0 / S))

    def _estimate_reliability(self, embedding: NarrativeEmbedding) -> float:
        B = self.params.bootstrap_samples
        dim_names = ["S", "N", "I", "Sy", "E", "Em"]
        ldri_samples = []
        rng = np.random.default_rng(42)
        for _ in range(B):
            eps = {}
            for d in dim_names:
                Hd = getattr(embedding, f"H_{d}", None)
                if Hd is not None and len(Hd) > 0:
                    noise = rng.normal(0, 0.01, size=len(Hd))
                    eps[d] = noise
            a_vals = []
            for d in dim_names:
                Hd = getattr(embedding, f"H_{d}", None)
                if Hd is None or len(Hd) == 0:
                    a_vals.append(0.0)
                else:
                    perturbed = Hd + eps.get(d, np.zeros_like(Hd))
                    norm = float(np.linalg.norm(perturbed))
                    a_d = 1.0 - math.exp(-norm ** 2 / (2.0 * self.params.sigma_max ** 2))
                    a_vals.append(a_d)
            R = np.linalg.norm(a_vals) / math.sqrt(6.0)
            C = self.coupling_factor(embedding.H_pairs + rng.normal(0, 0.01, size=len(embedding.H_pairs)))
            T = self.resonance_factor(embedding.H_res + rng.normal(0, 0.01, size=len(embedding.H_res)))
            raw = self._ldri_raw(R, (1+C)/2, (1+T)/2, 0.75)
            ldri_samples.append(raw)
        if len(ldri_samples) < 2:
            return 0.5
        sigma_ldri = float(np.std(ldri_samples))
        rel = 1.0 - min(1.0, sigma_ldri / 0.1)
        return float(rel)

    def _decompose(self, R_signal: float, C_tilde: float,
                   T_tilde: float, M_tilde: float,
                   Gamma: float, raw: float, ldri: float) -> tuple[dict[str, float], dict[str, float]]:
        p = self.params
        S = p.alpha + p.beta + p.gamma + p.delta
        C = 2.0 * C_tilde - 1.0
        T = 2.0 * T_tilde - 1.0
        M = 2.0 * M_tilde - 1.0

        c_int = (p.alpha / S) * Gamma * math.log(max(1e-10, (1.0 + R_signal) / max(1e-10, 1.0 - R_signal)))
        c_intr = (p.beta / S) * Gamma * math.log(max(1.0, 1.0 + C))
        c_temp = (p.gamma / S) * Gamma * math.log(max(1.0, 1.0 + T))
        c_res = (p.gamma / S) * Gamma * math.exp(-1.0 / 0.3) / (1.0 - math.exp(-1.0 / 0.3)) * min(1.0, T)
        c_mem = (p.delta / S) * Gamma * math.log(max(1.0, 1.0 + M))
        c_unc = (1.0 - Gamma) * raw

        components = {
            "intrinsic": float(c_int),
            "interaction": float(c_intr),
            "temporal": float(c_temp),
            "resonance": float(c_res),
            "memory": float(c_mem),
            "uncertainty": float(c_unc),
        }
        total = ldri + c_unc + 1e-10
        fractions = {
            k: float(v / total) for k, v in components.items()
        }
        return components, fractions
