"""
Domain models for Karachi real estate data.
These are used both as Pydantic schemas (API in/out) and as structured
records stored in Pinecone metadata and the LightRAG graph.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class FileStatus(str, Enum):
    allotted = "allotted"
    transferred = "transferred"
    on_hold = "on_hold"
    disputed = "disputed"
    cancelled = "cancelled"
    possession_given = "possession_given"
    under_litigation = "under_litigation"


class Society(str, Enum):
    dha_city_karachi = "dha_city_karachi"
    dha_karachi = "dha_karachi"
    bahria_town_karachi = "bahria_town_karachi"
    malir_city = "malir_city"
    malir_town_residencia = "malir_town_residencia"
    saadi_town = "saadi_town"
    gulshan_e_iqbal = "gulshan_e_iqbal"
    npha_karachi = "npha_karachi"
    other = "other"


class PlotRecord(BaseModel):
    """A property/plot file as registered with its issuing authority."""

    file_number: str = Field(..., description="Official file number (e.g. DCK-S14-0042)")
    society: Society
    sector_or_precinct: str = Field(..., description="Sector (DCK) or Precinct (BTK) or Block (MCDA)")
    plot_size_sqyd: int = Field(..., description="Plot size in square yards")
    plot_type: str = Field(default="residential", description="residential | commercial | farmhouse")
    owner_cnic: Optional[str] = None
    owner_name: Optional[str] = None
    file_status: FileStatus = FileStatus.allotted
    allotment_date: Optional[date] = None
    metadata: dict[str, Any] = {}


class DutyRecord(BaseModel):
    """SRB stamp duty and CVT record for a property transaction."""

    challan_number: str = Field(..., description="SRB challan receipt number")
    file_number: str = Field(..., description="Property file this duty record relates to")
    transaction_date: date
    declared_value_pkr: float
    fbr_valuation_pkr: float
    taxable_value_pkr: float = Field(..., description="Higher of declared vs FBR")
    stamp_duty_rate_pct: float = Field(..., description="Rate applied at time of transaction (e.g. 3.0, 4.0, 5.0)")
    stamp_duty_amount_pkr: float
    cvt_rate_pct: float
    cvt_amount_pkr: float
    wht_seller_pct: float = Field(default=1.0, description="Section 236C withholding")
    wht_buyer_pct: float = Field(default=1.0, description="Section 236K advance tax")
    finance_act_year: int = Field(default=2024, description="Which Finance Act rates were applied")
    srb_branch: Optional[str] = None
    notes: Optional[str] = None


class PriceIndex(BaseModel):
    """Quarterly price-per-sqyd index entry for a specific society and plot category."""

    society: Society
    sector_or_precinct: Optional[str] = None
    plot_size_sqyd: Optional[int] = None
    quarter: str = Field(..., description="Quarter in YYYY-QN format, e.g. 2024-Q2")
    price_per_sqyd_pkr: float
    source: str = Field(..., description="zameen | graana | manual | srb_transactions")
    index_value: Optional[float] = Field(None, description="Normalized index (Q1-2023=100)")
    recorded_date: Optional[date] = None
    notes: Optional[str] = None


class PossessionRecord(BaseModel):
    """Tracks whether possession of a plot has been handed over."""

    file_number: str
    society: Society
    sector_or_precinct: str
    possession_given: bool = False
    possession_date: Optional[date] = None
    conditions_met: list[str] = Field(
        default_factory=list,
        description="Conditions satisfied before possession: e.g. 'DC paid', 'No litigation'"
    )
    conditions_pending: list[str] = Field(
        default_factory=list,
        description="Conditions not yet met: e.g. 'Gas not available', 'Water pending'"
    )
    builder_signoff: bool = False
    notes: Optional[str] = None


class LitigationFlag(BaseModel):
    """Court case or legal dispute associated with a property file."""

    file_number: str
    society: Society
    court_name: str = Field(..., description="SHC | Supreme Court | Civil Court | Cantonment Tribunal | DHA IDC")
    case_number: str
    case_title: Optional[str] = None
    filing_date: Optional[date] = None
    status: str = Field(..., description="active | resolved | stayed | dismissed | appeal_pending")
    restriction: str = Field(
        ...,
        description="What the case restricts: e.g. 'no transfer', 'no possession', 'no construction'"
    )
    resolution_date: Optional[date] = None
    notes: Optional[str] = None


class DigitalNOC(BaseModel):
    """SBCA Digital NOC record (introduced 2024)."""

    noc_number: str
    file_number: Optional[str] = None
    society: Optional[Society] = None
    noc_type: str = Field(..., description="building_permission | completion_certificate | transfer_noc")
    issued_date: date
    expiry_date: date
    verification_url: str = Field(..., description="https://sbca.gos.pk/noc-verify/{noc_number}")
    qr_code_data: Optional[str] = None
    issuing_officer: Optional[str] = None
    is_valid: bool = True


# ── Request/Response models for property endpoints ────────────────────────────

class PossessionStatusRequest(BaseModel):
    society: Society
    sector_or_precinct: Optional[str] = None


class PossessionStatusResponse(BaseModel):
    society: str
    sector_or_precinct: Optional[str]
    total_plots: int
    possession_given: int
    possession_pending: int
    pct_complete: float
    pending_reasons: list[str]
    as_of: str


class PriceHistoryRequest(BaseModel):
    society: Society
    sector_or_precinct: Optional[str] = None
    plot_size_sqyd: Optional[int] = None
    from_quarter: str = Field(..., description="YYYY-QN e.g. 2023-Q1")
    to_quarter: str = Field(..., description="YYYY-QN e.g. 2025-Q1")


class PriceHistoryResponse(BaseModel):
    society: str
    sector_or_precinct: Optional[str]
    plot_size_sqyd: Optional[int]
    data_points: list[PriceIndex]
    min_price_sqyd: float
    max_price_sqyd: float
    latest_price_sqyd: float
    change_pct: float = Field(..., description="% change from from_quarter to to_quarter")


class LitigationCheckResponse(BaseModel):
    file_number: str
    has_litigation: bool
    flags: list[LitigationFlag]
    restriction_summary: str
    safe_to_transfer: bool


class DutyCalculatorRequest(BaseModel):
    society: Society
    declared_value_pkr: float
    fbr_valuation_pkr: float
    property_type: str = "residential"
    buyer_is_filer: bool = True
    seller_is_filer: bool = True
    transaction_date: Optional[date] = None


class DutyCalculatorResponse(BaseModel):
    taxable_value_pkr: float
    stamp_duty_rate_pct: float
    stamp_duty_pkr: float
    cvt_rate_pct: float
    cvt_pkr: float
    wht_seller_pkr: float
    wht_buyer_pkr: float
    total_tax_pkr: float
    finance_act_applied: int
    breakdown: dict[str, float]
