# LIT-DISC Configuration Manager
# Reference: Architecture Specification §7

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CouplingComponentParams:
    mu: float
    omega: float
    alpha: float
    beta_i: float


@dataclass(frozen=True)
class CouplingParams:
    sigma_tau: float = 0.1
    C_sim: CouplingComponentParams = field(default_factory=lambda: CouplingComponentParams(0.15, 0.30, 0.10, 0.20))
    C_prox: CouplingComponentParams = field(default_factory=lambda: CouplingComponentParams(0.55, 0.50, 0.10, 0.20))
    C_conf: CouplingComponentParams = field(default_factory=lambda: CouplingComponentParams(0.50, 1.00, 0.20, 0.10))
    C_comp: CouplingComponentParams = field(default_factory=lambda: CouplingComponentParams(0.10, 0.20, 0.05, 0.15))
    C_sym: CouplingComponentParams = field(default_factory=lambda: CouplingComponentParams(0.60, 0.40, 0.05, 0.25))
    C_rel: CouplingComponentParams = field(default_factory=lambda: CouplingComponentParams(0.85, 0.30, 0.05, 0.20))


@dataclass(frozen=True)
class MemoryParams:
    key_dim: int = 32
    sigma_kernel: float = 1.0
    seed_W: int = 0
    seed_b: int = 1
    theta_store: float = 0.5
    theta_retrieve: float = 0.7
    theta_forget: float = 0.01
    rho_reinforce: float = 0.1
    delta_decay: float = 0.05


@dataclass(frozen=True)
class ResonanceParams:
    lambda_decay: float = 0.3
    eta_resonance: float = 0.2


@dataclass(frozen=True)
class EmbeddingParams:
    dimension: int = 120


@dataclass(frozen=True)
class LDRIParams:
    alpha: float = 2.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    sigma_max: float = 0.1
    bootstrap_samples: int = 100


@dataclass(frozen=True)
class HMPDTConfig:
    refinement_iterations: int = 5
    beta_salience: float = 2.0
    beta_topdown: float = 0.3
    epsilon_convergence: float = 1e-6
    narrative_samples: int = 100
    coupling: CouplingParams = field(default_factory=CouplingParams)
    memory: MemoryParams = field(default_factory=MemoryParams)
    resonance: ResonanceParams = field(default_factory=ResonanceParams)
    embedding: EmbeddingParams = field(default_factory=EmbeddingParams)


DEFAULT_LDRI_PARAMS = LDRIParams()


@dataclass(frozen=True)
class LITDISCConfig:
    hmp_dt: HMPDTConfig = field(default_factory=HMPDTConfig)
    ldri: LDRIParams = field(default_factory=LDRIParams)
    alternate_parameter_sets: dict[str, Any] = field(default_factory=dict)
