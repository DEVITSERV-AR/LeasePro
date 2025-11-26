import streamlit as st
import pandas as pd
from io import BytesIO

# ------------ Helper functions ------------

def pmt(rate, nper, pv, fv=0.0):
    """
    Correct Excel-style PMT for end-of-period payments.
    rate = rate per period
    nper = number of periods
    pv   = present value (positive)
    fv   = future value (residual inflow at end, positive)
    Returns a negative payment (outflow), like Excel PMT.
    """
    if nper == 0:
        raise ValueError("nper must be > 0")
    if rate == 0:
        return -(pv - fv) / nper
    factor = (1 + rate) ** nper
    return -(rate * (pv * factor - fv) / (factor - 1))


def npv(rate, cashflows):
    """
    Net Present Value for equally spaced cashflows at t = 0,1,2,...
    """
    total = 0.0
    for t, cf in enumerate(cashflows):
        total += cf / ((1 + rate) ** t)
    return total


def compute_irr(cashflows, low=-0.9999, high=10.0, tol=1e-7, max_iter=200):
    """
    Simple IRR using bisection.
    Returns None if no sign change in NPV within [low, high].
    """
    cfs = list(cashflows)
    has_pos = any(cf > 0 for cf in cfs)
    has_neg = any(cf < 0 for cf in cfs)
    if not (has_pos and has_neg):
        return None

    npv_low = npv(low, cfs)
    npv_high = npv(high, cfs)
    if npv_low * npv_high > 0:
        return None

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        npv_mid = npv(mid, cfs)
        if abs(npv_mid) < tol:
            return mid
        if npv_low * npv_mid < 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid

    return (low + high) / 2.0


# ------------ Streamlit app config ------------

st.set_page_config(page_title="LeasePro - Lease Calculator", layout="wide")

page = st.sidebar.radio("Navigation", ["Input Page", "Output Page"])

if "inputs" not in st.session_state:
    st.session_state.inputs = {}

# ------------ PAGE 1: INPUTS ------------

if page == "Input Page":
    st.title("📘 LeasePro – Input Page")

    st.subheader("Basic Deal Parameters")
    col1, col2 = st.columns(2)

    with col1:
        asset_cost = st.number_input("Asset Cost", value=85000.0, step=1000.0)
        logistics = st.number_input("Logistics Cost", value=0.0, step=1000.0)
        insurance = st.number_input("Insurance Cost", value=0.0, step=1000.0)
        installation = st.number_input("Installation Cost", value=0.0, step=1000.0)

    with col2:
        down_payment = st.number_input("Down Payment (if any)", value=0.0, step=1000.0)
        lease_term = st.number_input("Lease Term (Months)", value=36, step=1)
        interest_rate = st.number_input("Target Annual Interest Rate (%)", value=14.0, step=0.25)
        residual_value = st.number_input("Residual Value (Inflow at End of Term)", value=8500.0, step=1000.0)

    st.subheader("Fees, Deposits & Structure")
    col3, col4 = st.columns(2)

    with col3:
        processing_fee = st.number_input("Upfront Processing Fee (received at start)", value=0.0, step=1000.0)
        security_deposit = st.number_input("Security Deposit (received at start)", value=0.0, step=1000.0)
        deposit_type = st.selectbox(
            "Security Deposit Type",
            ["None / Not applicable", "Refundable at end", "Non-refundable"]
        )
        payment_timing = st.selectbox(
            "Payment Timing",
            ["Arrears (End of month)", "Advance (Start of month)"]
        )

    with col4:
        payment_interval = st.selectbox(
            "Payment Interval",
            options=[1, 3],
            format_func=lambda x: "Monthly" if x == 1 else "Quarterly",
        )
        balloon_extra = st.number_input(
            "Balloon Extra Rental in Final Period (Excl GST)",
            value=0.0,
            step=1000.0,
            help="Additional extra rental in the last payment period (over and above normal rent)."
        )
        gst_rate = st.number_input("GST Rate (%)", value=18.0, step=1.0)
        gst_inside_emi = st.checkbox("Show EMI as GST-inclusive amount (customer view)", value=True)

    if st.button("Save & Go to Output Page"):
        st.session_state.inputs = {
            "asset_cost": float(asset_cost),
            "down_payment": float(down_payment),
            "logistics": float(logistics),
            "insurance": float(insurance),
            "installation": float(installation),
            "lease_term": int(lease_term),
            "interest_rate": float(interest_rate),
            "residual_value": float(residual_value),
            "gst_rate": float(gst_rate),
            "payment_interval": int(payment_interval),
            "processing_fee": float(processing_fee),
            "security_deposit": float(security_deposit),
            "deposit_type": deposit_type,
            "balloon_extra": float(balloon_extra),
            "gst_inside_emi": bool(gst_inside_emi),
            "payment_timing": payment_timing,
        }
        st.success("Inputs saved. Switch to 'Output Page' from the left.")

# ------------ PAGE 2: OUTPUTS ------------

if page == "Output Page":
    st.title("📗 LeasePro – Results")

    if not st.session_state.inputs:
        st.warning("Please fill the inputs on the 'Input Page' first.")
        st.stop()

    I = st.session_state.inputs

    # --- Core values ---
    total_invested = I["asset_cost"] + I["logistics"] + I["insurance"] + I["installation"]
    net_financing = total_invested - I["down_payment"]

    # --- Rate per period & number of periods ---
    if I["payment_interval"] == 1:       # Monthly
        rate_per_period = I["interest_rate"] / 1200.0
        n_periods = I["lease_term"]
    else:                                # Quarterly
        rate_per_period = I["interest_rate"] / 400.0
        n_periods = I["lease_term"] / 3.0

    # --- Base rental (arrears) using custom PMT ---
    raw_pmt_arrears = pmt(rate_per_period, n_periods, net_financing, I["residual_value"])
    rent_excl_arrears = -raw_pmt_arrears  # positive

    # --- Adjust for advance rentals (annuity due) ---
    if I["payment_timing"] == "Advance (Start of month)" and rate_per_period != 0:
        rent_excl = rent_excl_arrears / (1.0 + rate_per_period)
    else:
        rent_excl = rent_excl_arrears

    # --- Base GST & EMI per payment period ---
    gst_per_period = rent_excl * I["gst_rate"] / 100.0
    rent_incl = rent_excl + gst_per_period

    # --- PTPM / PTPQ (based on net financing) ---
    ptpm = ptpq = None    # PTPM (monthly) or PTPQ (quarterly)
    if net_financing != 0:
        if I["payment_interval"] == 1:
            ptpm = rent_excl / (net_financing / 1000.0)
        else:
            ptpq = rent_excl / (net_financing / 1000.0)

    # --- Amortisation schedule on monthly basis ---
    schedule_rows = []
    opening = net_financing
    monthly_rate = I["interest_rate"] / 1200.0

    for m in range(1, I["lease_term"] + 1):
        is_payment_month = (m % I["payment_interval"] == 0)

        payment_excl = 0.0
        interest = 0.0
        principal = 0.0

        if I["payment_timing"] == "Arrears (End of month)":
            # interest first on opening, then payment
            interest = opening * monthly_rate
            if is_payment_month:
                payment_excl = rent_excl
                if m == I["lease_term"]:
                    payment_excl += I["balloon_extra"]
            principal = payment_excl - interest
            closing = opening - principal

        else:  # Advance (Start of month)
            # payment at start of period (before interest)
            if is_payment_month:
                payment_excl = rent_excl
                if m == I["lease_term"]:
                    payment_excl += I["balloon_extra"]
            principal = payment_excl
            balance_after_payment = opening - principal
            # interest accrues on reduced balance
            interest = balance_after_payment * monthly_rate
            closing = balance_after_payment + interest

        gst_val = payment_excl * I["gst_rate"] / 100.0 if payment_excl != 0 else 0.0
        total_incl = payment_excl + gst_val

        # Residual treated as a separate inflow at final month
        residual_inflow = I["residual_value"] if m == I["lease_term"] else 0.0

        # Deposit refund if applicable
        deposit_refund = 0.0
        if I["deposit_type"] == "Refundable at end" and m == I["lease_term"]:
            deposit_refund = I["security_deposit"]

        total_incl_with_other = total_incl + residual_inflow + deposit_refund

        schedule_rows.append(
            [
                m,
                opening,
                payment_excl,
                interest,
                principal,
                closing,
                gst_val,
                total_incl,
                residual_inflow,
                deposit_refund,
                total_incl_with_other,
            ]
        )

        opening = closing

    df = pd.DataFrame(
        schedule_rows,
        columns=[
            "Month",
            "Opening",
            "Payment (Excl GST)",
            "Interest",
            "Principal",
            "Closing",
            "GST",
            "Total (Incl GST)",
            "Residual Inflow",
            "Deposit Refund",
            "Total Inflow (Incl GST + Residual + Deposit Refund)",
        ],
    )

    total_gst = float(df["GST"].sum())
    total_rent_incl = float(df["Total (Incl GST)"].sum())
    total_residual_inflow = float(df["Residual Inflow"].sum())
    total_deposit_refund = float(df["Deposit Refund"].sum())
    total_inflow_all = float(df["Total Inflow (Incl GST + Residual + Deposit Refund)"].sum())

    # --- IRR Calculations (lessor view) ---
    # Initial outflow: total invested minus down payment
    # Initial inflows: processing fee + security deposit
    cf0 = -total_invested + I["down_payment"] + I["processing_fee"] + I["security_deposit"]

    # Excl GST: rentals (excl GST) + residual + deposit refund
    cf_excl = [cf0] + list(df["Payment (Excl GST)"] + df["Residual Inflow"] + df["Deposit Refund"])

    # Incl GST: rentals incl GST + residual + deposit refund
    cf_incl = [cf0] + list(df["Total Inflow (Incl GST + Residual + Deposit Refund)"])

    mirr_excl = compute_irr(cf_excl)
    mirr_incl = compute_irr(cf_incl)

    # --- Summary display ---
    st.subheader("Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Invested Cost", f"{total_invested:,.2f}")
    c2.metric("Net Financing Amount", f"{net_financing:,.2f}")
    c3.metric("Residual Value (Inflow at End)", f"{I['residual_value']:,.2f}")

    c4, c5, c6 = st.columns(3)

    if I["gst_inside_emi"]:
        c4.metric("Customer EMI (Incl GST)", f"{rent_incl:,.2f}")
        c5.metric("Base Rental (Excl GST)", f"{rent_excl:,.2f}")
        c6.metric("GST Portion per EMI", f"{gst_per_period:,.2f}")
    else:
        c4.metric("Periodic Rental (Excl GST)", f"{rent_excl:,.2f}")
        c5.metric("Periodic GST", f"{gst_per_period:,.2f}")
        c6.metric("Periodic Rental (Incl GST)", f"{rent_incl:,.2f}")

    if ptpm is not None:
        st.metric("PTPM (Per Thousand Per Month)", f"{ptpm:,.4f}")
    if ptpq is not None:
        st.metric("PTPQ (Per Thousand Per Quarter)", f"{ptpq:,.4f}")

    st.metric("Total GST on Rentals", f"{total_gst:,.2f}")
    st.metric("Total Rentals (Incl GST, excl Residual/Deposit)", f"{total_rent_incl:,.2f}")
    st.metric("Residual Inflow (End)", f"{total_residual_inflow:,.2f}")
    st.metric("Deposit Refund (End)", f"{total_deposit_refund:,.2f}")
    st.metric("Total Inflow (Rentals + GST + Residual + Deposit Refund)", f"{total_inflow_all:,.2f}")

    if mirr_excl is not None:
        airr_excl = (1 + mirr_excl) ** 12 - 1
        st.write(f"**Monthly IRR (Excl GST):** {mirr_excl * 100:.4f}%")
        st.write(f"**Annualised IRR (Excl GST):** {airr_excl * 100:.4f}%")
    else:
        st.warning("Could not compute IRR (Excl GST) – cashflow pattern has no valid IRR in search range.")

    st.markdown("---")

    if mirr_incl is not None:
        airr_incl = (1 + mirr_incl) ** 12 - 1
        st.write(f"**Monthly IRR (Incl GST):** {mirr_incl * 100:.4f}%")
        st.write(f"**Annualised IRR (Incl GST):** {airr_incl * 100:.4f}%")
    else:
        st.warning("Could not compute IRR (Incl GST) – cashflow pattern has no valid IRR in search range.")

    # --- Amortisation table on screen ---
    st.subheader("Amortisation Schedule (Lessor View)")

    df_display = df.copy()
    for col in [
        "Opening",
        "Payment (Excl GST)",
        "Interest",
        "Principal",
        "Closing",
        "GST",
        "Total (Incl GST)",
        "Residual Inflow",
        "Deposit Refund",
        "Total Inflow (Incl GST + Residual + Deposit Refund)",
    ]:
        df_display[col] = df_display[col].round(2)

    st.dataframe(df_display, height=400, use_container_width=True)

    # --- Export to Excel ---
    st.subheader("Export")

    # Build a summary dataframe for Excel
    summary_data = {
        "Parameter": [
            "Asset Cost",
            "Logistics Cost",
            "Insurance Cost",
            "Installation Cost",
            "Total Invested Cost",
            "Down Payment",
            "Net Financing",
            "Lease Term (Months)",
            "Payment Interval (Months)",
            "Payment Timing",
            "Annual Interest Rate (%)",
            "Residual Value",
            "Processing Fee",
            "Security Deposit",
            "Deposit Type",
            "Balloon Extra (Excl GST)",
            "GST Rate (%)",
            "Base Rental (Excl GST)",
            "EMI (Incl GST)",
            "PTPM" if ptpm is not None else "PTPQ",
            "Total GST on Rentals",
            "Total Rentals Incl GST (excl Residual/Deposit)",
            "Residual Inflow (End)",
            "Deposit Refund (End)",
            "Total Inflow (All)",
            "Monthly IRR (Excl GST)",
            "Annual IRR (Excl GST)",
            "Monthly IRR (Incl GST)",
            "Annual IRR (Incl GST)",
        ],
        "Value": [
            I["asset_cost"],
            I["logistics"],
            I["insurance"],
            I["installation"],
            total_invested,
            I["down_payment"],
            net_financing,
            I["lease_term"],
            I["payment_interval"],
            I["payment_timing"],
            I["interest_rate"],
            I["residual_value"],
            I["processing_fee"],
            I["security_deposit"],
            I["deposit_type"],
            I["balloon_extra"],
            I["gst_rate"],
            rent_excl,
            rent_incl,
            ptpm if ptpm is not None else ptpq,
            total_gst,
            total_rent_incl,
            total_residual_inflow,
            total_deposit_refund,
            total_inflow_all,
            mirr_excl * 100 if mirr_excl is not None else None,
            ((1 + mirr_excl) ** 12 - 1) * 100 if mirr_excl is not None else None,
            mirr_incl * 100 if mirr_incl is not None else None,
            ((1 + mirr_incl) ** 12 - 1) * 100 if mirr_incl is not None else None,
        ],
    }

    summary_df = pd.DataFrame(summary_data)

    # Create Excel in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        df.to_excel(writer, sheet_name="Amortisation", index=False)
    output.seek(0)

    st.download_button(
        label="📥 Download Excel (Summary + Amortisation)",
        data=output,
        file_name="LeasePro_Lease_Calculation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
