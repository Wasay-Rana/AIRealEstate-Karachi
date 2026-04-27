"""
Property-domain API endpoints for Karachi real estate queries.
These endpoints answer structured questions directly from the knowledge base
rather than going through the full RAG pipeline.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.logging import get_logger
from app.models.property import (
    DutyCalculatorRequest,
    DutyCalculatorResponse,
    FileStatus,
    LitigationCheckResponse,
    LitigationFlag,
    PossessionRecord,
    PossessionStatusRequest,
    PossessionStatusResponse,
    PriceHistoryResponse,
    PriceIndex,
    Society,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/properties", tags=["properties"])


# ── Possession Status ─────────────────────────────────────────────────────────

# Static knowledge derived from the ingested corpus.
# In production this would be backed by a database populated from ingested records.
_DCK_POSSESSION = {
    "sector_1": PossessionRecord(
        file_number="ALL", society=Society.dha_city_karachi,
        sector_or_precinct="sector_1", possession_given=True,
        possession_date=date(2022, 6, 1),
        conditions_met=["DC paid", "No litigation", "Roads complete", "Water available"],
        conditions_pending=[],
    ),
    "sector_14": PossessionRecord(
        file_number="ALL", society=Society.dha_city_karachi,
        sector_or_precinct="sector_14", possession_given=True,
        possession_date=date(2024, 4, 15),
        conditions_met=["DC paid", "Roads graded", "Boundary wall complete"],
        conditions_pending=["Sui gas not yet available", "Hospital not yet operational"],
        notes="Possession given in April 2024 to approx 1,200 plot holders. Gas via LPG.",
    ),
    "sector_15": PossessionRecord(
        file_number="ALL", society=Society.dha_city_karachi,
        sector_or_precinct="sector_15", possession_given=False,
        conditions_met=["Balloting complete", "Development charges notified"],
        conditions_pending=["Roads not complete", "Utilities pending", "Possession estimated 2026"],
    ),
    "sector_16": PossessionRecord(
        file_number="ALL", society=Society.dha_city_karachi,
        sector_or_precinct="sector_16", possession_given=False,
        conditions_met=["Balloting complete"],
        conditions_pending=["Development in progress", "Possession estimated 2026"],
    ),
    "sector_17": PossessionRecord(
        file_number="ALL", society=Society.dha_city_karachi,
        sector_or_precinct="sector_17", possession_given=False,
        conditions_met=["Balloting complete"],
        conditions_pending=["Grading in progress", "Possession estimated late 2026"],
    ),
}

_DCK_SECTOR_TOTALS = {
    "sector_1": (800, 800), "sector_2": (950, 950), "sector_3": (900, 900),
    "sector_4": (870, 850), "sector_5": (820, 810), "sector_6": (760, 745),
    "sector_7": (680, 660), "sector_8": (590, 570),
    "sector_9": (1100, 200), "sector_10": (1250, 180), "sector_11": (1300, 120),
    "sector_12": (980, 80), "sector_13": (1050, 60),
    "sector_14": (1400, 1200), "sector_15": (1500, 0), "sector_16": (1350, 0),
    "sector_17": (1200, 0), "sector_18": (900, 0), "sector_19": (750, 0),
    "sector_20": (600, 0),
}

_BTK_POSSESSION = {
    "precinct_1": (True, "Possession given 2018+"),
    "precinct_7": (True, "Possession given 2019+"),
    "precinct_19": (True, "Possession given 2020; some units still pending dues"),
    "precinct_25": (True, "Possession given 2021"),
    "precinct_27": (True, "Possession given 2021"),
    "precinct_32": (True, "Possession given 2022–2023 in batches"),
    "precinct_35": (True, "Golf City possession 2023–2024"),
    "precinct_37": (False, "Phase 2 Extension — booking stage only"),
    "precinct_38": (False, "Phase 2 Extension — booking stage only"),
    "precinct_40": (False, "Phase 2 Extension — booking stage only"),
}


@router.get("/possession-status", response_model=PossessionStatusResponse)
async def possession_status(
    society: Society = Query(..., description="Society/scheme name"),
    sector_or_precinct: Optional[str] = Query(None, description="Sector number (DCK) or Precinct (BTK)"),
) -> PossessionStatusResponse:
    """
    Returns possession handover status breakdown for a given society and optional sector/precinct.
    """
    society_label = society.value.replace("_", " ").title()

    if society == Society.dha_city_karachi:
        if sector_or_precinct:
            key = sector_or_precinct.lower().replace(" ", "_")
            if not key.startswith("sector_"):
                key = f"sector_{key}"
            totals = _DCK_SECTOR_TOTALS.get(key)
            rec = _DCK_POSSESSION.get(key)
            if totals is None:
                raise HTTPException(status_code=404, detail=f"Sector '{sector_or_precinct}' not found in DCK records")
            total, given = totals
            pending_reasons = rec.conditions_pending if rec else []
        else:
            total = sum(t for t, _ in _DCK_SECTOR_TOTALS.values())
            given = sum(g for _, g in _DCK_SECTOR_TOTALS.values())
            pending_reasons = [
                "Sectors 9–13: Development in progress; utilities pending",
                "Sectors 15–20: Under development; possession est. 2026+",
                "Sui gas not available in sectors 14–20",
            ]

        return PossessionStatusResponse(
            society=society_label,
            sector_or_precinct=sector_or_precinct,
            total_plots=total,
            possession_given=given,
            possession_pending=total - given,
            pct_complete=round(given / total * 100, 1) if total > 0 else 0.0,
            pending_reasons=pending_reasons,
            as_of="Q1 2025",
        )

    if society == Society.bahria_town_karachi:
        if sector_or_precinct:
            key = sector_or_precinct.lower().replace(" ", "_")
            if not key.startswith("precinct_"):
                key = f"precinct_{key}"
            info = _BTK_POSSESSION.get(key)
            if info is None:
                raise HTTPException(status_code=404, detail=f"Precinct '{sector_or_precinct}' not found in BTK records")
            given_flag, note = info
            return PossessionStatusResponse(
                society=society_label,
                sector_or_precinct=sector_or_precinct,
                total_plots=500,
                possession_given=500 if given_flag else 0,
                possession_pending=0 if given_flag else 500,
                pct_complete=100.0 if given_flag else 0.0,
                pending_reasons=[] if given_flag else [note],
                as_of="Q1 2025",
            )
        # Aggregate BTK
        total_given = sum(1 for v, _ in _BTK_POSSESSION.values() if v)
        pct = round(total_given / len(_BTK_POSSESSION) * 100, 1)
        return PossessionStatusResponse(
            society=society_label,
            sector_or_precinct=None,
            total_plots=len(_BTK_POSSESSION) * 500,
            possession_given=total_given * 500,
            possession_pending=(len(_BTK_POSSESSION) - total_given) * 500,
            pct_complete=pct,
            pending_reasons=["Precincts 37–45 (Phase 2 Extension) in booking stage only"],
            as_of="Q1 2025",
        )

    raise HTTPException(status_code=422, detail=f"Possession data not yet available for society: {society.value}")


# ── Price History ─────────────────────────────────────────────────────────────

_PRICE_DATA: dict[str, list[tuple[str, float]]] = {
    "dha_city_karachi|sector_1_8|*": [
        ("2023-Q1", 14000), ("2023-Q2", 15200), ("2023-Q3", 16800), ("2023-Q4", 18500),
        ("2024-Q1", 22000), ("2024-Q2", 27500), ("2024-Q3", 30000), ("2024-Q4", 32500),
        ("2025-Q1", 34000),
    ],
    "dha_city_karachi|sector_14_17|*": [
        ("2023-Q1", 10500), ("2023-Q2", 11200), ("2023-Q3", 12000), ("2023-Q4", 13500),
        ("2024-Q1", 15500), ("2024-Q2", 19000), ("2024-Q3", 21500), ("2024-Q4", 23000),
        ("2025-Q1", 24500),
    ],
    "bahria_town_karachi|precinct_7|125": [
        ("2023-Q1", 38000), ("2023-Q2", 42000), ("2023-Q3", 45000), ("2023-Q4", 48000),
        ("2024-Q1", 50500), ("2024-Q2", 52000), ("2024-Q3", 54500), ("2024-Q4", 57000),
        ("2025-Q1", 60000),
    ],
    "bahria_town_karachi|precinct_19|250": [
        ("2023-Q1", 30000), ("2023-Q2", 32500), ("2023-Q3", 34000), ("2023-Q4", 34500),
        ("2024-Q1", 33000), ("2024-Q2", 30000), ("2024-Q3", 31500), ("2024-Q4", 33500),
        ("2025-Q1", 35500),
    ],
    "bahria_town_karachi|precinct_35|500": [
        ("2023-Q1", 48000), ("2023-Q2", 52000), ("2023-Q3", 55000), ("2023-Q4", 58000),
        ("2024-Q1", 60000), ("2024-Q2", 62500), ("2024-Q3", 65000), ("2024-Q4", 68000),
        ("2025-Q1", 72000),
    ],
    "malir_city|block_a_g|120": [
        ("2023-Q1", 25000), ("2023-Q2", 26500), ("2023-Q3", 28000), ("2023-Q4", 29500),
        ("2024-Q1", 31000), ("2024-Q2", 33500), ("2024-Q3", 35000), ("2024-Q4", 36500),
        ("2025-Q1", 38000),
    ],
    "malir_city|block_h_n|120": [
        ("2023-Q1", 20000), ("2023-Q2", 21500), ("2023-Q3", 22500), ("2023-Q4", 23500),
        ("2024-Q1", 25000), ("2024-Q2", 27000), ("2024-Q3", 28500), ("2024-Q4", 30000),
        ("2025-Q1", 31500),
    ],
    "saadi_town|*|120": [
        ("2023-Q1", 38000), ("2023-Q2", 40000), ("2023-Q3", 41500), ("2023-Q4", 42500),
        ("2024-Q1", 44000), ("2024-Q2", 46000), ("2024-Q3", 47500), ("2024-Q4", 49000),
        ("2025-Q1", 50500),
    ],
}


def _quarter_to_num(q: str) -> int:
    """Convert '2024-Q2' to sortable int 20242."""
    m = re.match(r"(\d{4})-Q([1-4])", q)
    if not m:
        raise ValueError(f"Invalid quarter format: {q!r}. Expected YYYY-QN")
    return int(m.group(1)) * 10 + int(m.group(2))


def _find_price_series(society: Society, sector_or_precinct: Optional[str], plot_size: Optional[int]) -> list[tuple[str, float]]:
    """Fuzzy-match against _PRICE_DATA keys."""
    soc = society.value
    sec = (sector_or_precinct or "*").lower().replace(" ", "_")
    size = str(plot_size) if plot_size else "*"

    # Try exact key first
    for key, data in _PRICE_DATA.items():
        k_soc, k_sec, k_size = key.split("|")
        if k_soc != soc:
            continue
        sec_match = k_sec == "*" or k_sec == sec or sec in k_sec or k_sec in sec
        size_match = k_size == "*" or k_size == size
        if sec_match and size_match:
            return data

    return []


@router.get("/price-history", response_model=PriceHistoryResponse)
async def price_history(
    society: Society = Query(...),
    sector_or_precinct: Optional[str] = Query(None),
    plot_size_sqyd: Optional[int] = Query(None),
    from_quarter: str = Query(default="2023-Q1", description="YYYY-QN"),
    to_quarter: str = Query(default="2025-Q1", description="YYYY-QN"),
) -> PriceHistoryResponse:
    """
    Returns quarterly price-per-sqyard trend for a society, optionally filtered by
    sector/precinct and plot size.
    """
    try:
        from_num = _quarter_to_num(from_quarter)
        to_num = _quarter_to_num(to_quarter)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    series = _find_price_series(society, sector_or_precinct, plot_size_sqyd)
    if not series:
        raise HTTPException(
            status_code=404,
            detail=f"No price data found for society={society.value}, sector/precinct={sector_or_precinct}, size={plot_size_sqyd}"
        )

    filtered = [
        PriceIndex(
            society=society,
            sector_or_precinct=sector_or_precinct,
            plot_size_sqyd=plot_size_sqyd,
            quarter=q,
            price_per_sqyd_pkr=p,
            source="zameen_quarterly_index",
        )
        for q, p in series
        if from_num <= _quarter_to_num(q) <= to_num
    ]

    if not filtered:
        raise HTTPException(status_code=404, detail="No data points in the requested quarter range")

    prices = [d.price_per_sqyd_pkr for d in filtered]
    change = round((prices[-1] - prices[0]) / prices[0] * 100, 1) if len(prices) > 1 else 0.0

    return PriceHistoryResponse(
        society=society.value,
        sector_or_precinct=sector_or_precinct,
        plot_size_sqyd=plot_size_sqyd,
        data_points=filtered,
        min_price_sqyd=min(prices),
        max_price_sqyd=max(prices),
        latest_price_sqyd=prices[-1],
        change_pct=change,
    )


# ── Litigation Check ──────────────────────────────────────────────────────────

# Populated from ingested legal records; demo data below
_LITIGATION_DB: dict[str, list[LitigationFlag]] = {
    "DCK-S09-0121": [
        LitigationFlag(
            file_number="DCK-S09-0121",
            society=Society.dha_city_karachi,
            court_name="SHC",
            case_number="SHC-2023-1045",
            case_title="Ahmed vs DHA City Karachi",
            filing_date=date(2023, 3, 12),
            status="active",
            restriction="no transfer",
            notes="Dual allotment dispute — two parties claim same plot. DHA investigating.",
        )
    ],
    "BTK-P15-0872": [
        LitigationFlag(
            file_number="BTK-P15-0872",
            society=Society.bahria_town_karachi,
            court_name="SHC",
            case_number="SHC-2022-4412",
            case_title="Sindh Government vs Bahria Town (land acquisition)",
            filing_date=date(2022, 8, 5),
            status="stayed",
            restriction="no construction",
            notes="Perimeter land dispute; construction stay until case resolved.",
        )
    ],
}


@router.get("/litigation-check", response_model=LitigationCheckResponse)
async def litigation_check(
    file_number: str = Query(..., description="Plot file number e.g. DCK-S14-0042"),
) -> LitigationCheckResponse:
    """
    Checks whether a property file has any active litigation flags.
    Returns restriction details and whether the file is safe to transfer.
    """
    flags = _LITIGATION_DB.get(file_number.upper(), [])
    active_flags = [f for f in flags if f.status in ("active", "stayed", "appeal_pending")]
    has_litigation = len(active_flags) > 0

    if not has_litigation:
        summary = "No active litigation found. File appears safe for transfer."
        safe = True
    else:
        restrictions = list({f.restriction for f in active_flags})
        summary = f"LITIGATION ALERT: {len(active_flags)} active flag(s). Restrictions: {'; '.join(restrictions)}"
        safe = not any("transfer" in f.restriction for f in active_flags)

    return LitigationCheckResponse(
        file_number=file_number.upper(),
        has_litigation=has_litigation,
        flags=active_flags,
        restriction_summary=summary,
        safe_to_transfer=safe,
    )


# ── Stamp Duty Calculator ─────────────────────────────────────────────────────

def _calculate_duty(
    taxable_value: float,
    property_type: str,
    is_cantonment: bool,
    buyer_filer: bool,
    seller_filer: bool,
    transaction_date: Optional[date],
) -> tuple[float, float, float, float, float, float, int]:
    """Returns (stamp_rate, stamp_duty, cvt_rate, cvt, wht_seller, wht_buyer, finance_act_year)."""
    # Determine applicable Finance Act year
    cutoff = date(2024, 7, 1)
    tx_date = transaction_date or date.today()
    finance_act = 2024 if tx_date >= cutoff else 2022

    # Stamp duty (Sindh Finance Act 2024)
    if finance_act >= 2024:
        if taxable_value <= 5_000_000:
            stamp_rate = 3.0
        elif taxable_value <= 20_000_000:
            stamp_rate = 4.0
        else:
            stamp_rate = 5.0
    else:
        stamp_rate = 3.0 if taxable_value <= 3_000_000 else 4.0

    stamp_duty = taxable_value * stamp_rate / 100

    # CVT
    if is_cantonment:
        cvt_rate = 1.0
    elif property_type == "commercial":
        cvt_rate = 4.0
    else:
        cvt_rate = 2.0
    cvt = taxable_value * cvt_rate / 100

    # WHT
    wht_seller = taxable_value * (1.0 if seller_filer else 2.0) / 100
    wht_buyer = taxable_value * (1.0 if buyer_filer else 2.0) / 100

    return stamp_rate, stamp_duty, cvt_rate, cvt, wht_seller, wht_buyer, finance_act


@router.post("/duty-calculator", response_model=DutyCalculatorResponse)
async def duty_calculator(request: DutyCalculatorRequest) -> DutyCalculatorResponse:
    """
    Calculates stamp duty, CVT, and withholding tax for a Karachi property transaction
    using Sindh Finance Act 2024 rates.
    """
    taxable = max(request.declared_value_pkr, request.fbr_valuation_pkr)
    is_cantonment = request.society in (
        Society.malir_city, Society.malir_town_residencia
    )

    stamp_rate, stamp_duty, cvt_rate, cvt, wht_s, wht_b, fa_year = _calculate_duty(
        taxable_value=taxable,
        property_type=request.property_type,
        is_cantonment=is_cantonment,
        buyer_filer=request.buyer_is_filer,
        seller_filer=request.seller_is_filer,
        transaction_date=request.transaction_date,
    )

    total = stamp_duty + cvt + wht_s + wht_b

    return DutyCalculatorResponse(
        taxable_value_pkr=taxable,
        stamp_duty_rate_pct=stamp_rate,
        stamp_duty_pkr=round(stamp_duty, 2),
        cvt_rate_pct=cvt_rate,
        cvt_pkr=round(cvt, 2),
        wht_seller_pkr=round(wht_s, 2),
        wht_buyer_pkr=round(wht_b, 2),
        total_tax_pkr=round(total, 2),
        finance_act_applied=fa_year,
        breakdown={
            "taxable_value": taxable,
            "stamp_duty": round(stamp_duty, 2),
            "cvt": round(cvt, 2),
            "wht_seller_236C": round(wht_s, 2),
            "wht_buyer_236K": round(wht_b, 2),
            "total": round(total, 2),
        },
    )
