"""Tests for aerodynamics.py.

Rocket parts, fins, etc. are replaced with lightweight stand-ins
(SimpleNamespace) so these tests don't depend on components.py behaviour.
`from components import *` inside aerodynamics.py must still be importable.

Run from the folder containing aerodynamics.py:
    python -m pytest -v
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from aerodynamics import aerodynamics as Aero

AREF = 0.01


# --------------------------------------------------------------------------
# Helpers / fixtures
# --------------------------------------------------------------------------

def make_aero(mach=0.5, alpha=0.05, **kw):
    return Aero(AREF, mach, alpha, kw.pop("kv", 1.5e-5), kw.pop("temp", 288.0), **kw)


@pytest.fixture
def aero():
    return make_aero()


@pytest.fixture
def cylinder():
    """Plain body tube with no area change; r(x) is constant."""
    return SimpleNamespace(radius=0.05, length=1.0, r=lambda x: 0.05 + 0.0 * x)


@pytest.fixture
def cone():
    """Transition-like part with an area change."""
    return SimpleNamespace(
        radius=0.05, length=0.3, area_0=0.0, area_l=0.0078, volume=0.0006,
    )


@pytest.fixture
def fin():
    return SimpleNamespace(
        span=0.1, area=0.005, Lc=0.0, root_chord=0.2, tip_chord=0.1,
        sweep_length=0.1, ar=2.0, c_hat=0.15, thickness=0.003, location=0.9,
    )


# Class names matter: cd_total dispatches on __class__.__name__
class nosecone_vk(SimpleNamespace):
    pass


class body_tube(SimpleNamespace):
    pass


class boattail(SimpleNamespace):
    pass


# --------------------------------------------------------------------------
# __init__
# --------------------------------------------------------------------------

def test_init_stores_attributes():
    a = Aero(0.02, 0.8, 0.1, 1.5e-5, 300.0, gamma=1.3, R_const=300)
    assert (a.Aref, a.mach, a.alpha, a.kv, a.temp) == (0.02, 0.8, 0.1, 1.5e-5, 300.0)
    assert (a.gamma, a.R_const) == (1.3, 300)


def test_init_defaults():
    a = make_aero()
    assert a.gamma == 1.4
    assert a.R_const == 287


# --------------------------------------------------------------------------
# Normal force
# --------------------------------------------------------------------------

def test_C_na_missing_area_returns_zero(aero):
    assert aero.C_na(SimpleNamespace()) == 0.0


def test_C_na_no_area_change_returns_zero(aero):
    assert aero.C_na(SimpleNamespace(area_0=0.005, area_l=0.005)) == 0.0


def test_C_na_zero_alpha():
    a = make_aero(alpha=0.0)
    body = SimpleNamespace(area_0=0.001, area_l=0.003)
    assert a.C_na(body) == pytest.approx(2 * 0.002 / AREF)


def test_C_na_nonzero_alpha(aero):
    body = SimpleNamespace(area_0=0.001, area_l=0.003)
    expected = 2 * 0.002 * math.sin(0.05) / (0.05 * AREF)
    assert aero.C_na(body) == pytest.approx(expected)


def test_C_na_small_alpha_approaches_zero_alpha_limit():
    body = SimpleNamespace(area_0=0.001, area_l=0.003)
    tiny = make_aero(alpha=1e-6).C_na(body)
    zero = make_aero(alpha=0.0).C_na(body)
    assert tiny == pytest.approx(zero, rel=1e-6)


def test_C_n_lift_missing_geometry_returns_zero(aero):
    assert aero.C_n_lift(SimpleNamespace()) == 0.0


@pytest.mark.parametrize("radius,length", [(0.05, 0.0), (0.0, 1.0), (-1, 1), (0.05, -1)])
def test_C_n_lift_nonpositive_dimensions_return_zero(aero, radius, length):
    assert aero.C_n_lift(SimpleNamespace(radius=radius, length=length)) == 0.0


def test_C_n_lift_formula(aero, cylinder):
    expected = 1.1 * 2 * 0.05 * 1.0 * math.sin(0.05) ** 2 / AREF
    assert aero.C_n_lift(cylinder) == pytest.approx(expected)


def test_C_n_lift_scales_with_K(aero, cylinder):
    assert aero.C_n_lift(cylinder, K=2.2) == pytest.approx(2 * aero.C_n_lift(cylinder, K=1.1))


def test_C_n_lift_zero_at_zero_alpha(cylinder):
    assert make_aero(alpha=0.0).C_n_lift(cylinder) == pytest.approx(0.0)


def test_C_na_fins_missing_geometry_returns_zero(aero):
    assert aero.C_na_fins(SimpleNamespace(), SimpleNamespace()) == 0.0


def test_C_na_fins_known_value_subsonic(fin):
    # mach=0 -> beta=1, Lc=0 -> hand-calculated value for span=0.1, area=0.005
    a = make_aero(mach=0.0)
    assert a.C_na_fins(fin, SimpleNamespace()) == pytest.approx(3.883222, rel=1e-5)


def test_C_na_fins_supersonic_positive(fin):
    a = make_aero(mach=2.0)
    assert a.C_na_fins(fin, SimpleNamespace()) > 0


def test_C_na_fins_tube_correction(aero, fin):
    no_tube = aero.C_na_fins(fin, SimpleNamespace())
    tube = SimpleNamespace(radius=0.05, span=0.1)
    kt = 1.0 + 0.05 / (0.1 + 0.05)
    assert aero.C_na_fins(fin, tube) == pytest.approx(no_tube * kt)


def test_C_n_total_empty_rocket(aero):
    assert aero.C_n_total(SimpleNamespace(parts=[])) == 0.0


def test_C_n_total_cylinder_uses_K_1_1(aero, cylinder):
    rocket = SimpleNamespace(parts=[cylinder])
    assert aero.C_n_total(rocket) == pytest.approx(aero.C_n_lift(cylinder, K=1.1))


def test_C_n_total_equal_areas_uses_default_K(aero, cylinder):
    cylinder.area_0 = cylinder.area_l = 0.0078
    rocket = SimpleNamespace(parts=[cylinder])
    assert aero.C_n_total(rocket) == pytest.approx(aero.C_n_lift(cylinder, K=1.1))


def test_C_n_total_cone_adds_area_change_and_lift(aero, cone):
    rocket = SimpleNamespace(parts=[cone])
    expected = aero.C_na(cone) * aero.alpha + aero.C_n_lift(cone, K=1.1)
    assert aero.C_n_total(rocket) == pytest.approx(expected)


def test_C_n_total_includes_fins(aero, cylinder, fin):
    tube = SimpleNamespace(radius=0.05, span=0.1)
    without = aero.C_n_total(SimpleNamespace(parts=[cylinder]))
    with_fins = aero.C_n_total(SimpleNamespace(parts=[cylinder], fins=fin, fin_tube=tube))
    assert with_fins == pytest.approx(without + aero.C_na_fins(fin, tube) * aero.alpha)


# --------------------------------------------------------------------------
# Moments
# --------------------------------------------------------------------------

def test_C_ma_missing_area_returns_zero(aero):
    assert aero.C_ma(SimpleNamespace()) == 0.0


def test_C_ma_no_area_change_returns_zero(aero):
    assert aero.C_ma(SimpleNamespace(area_0=1.0, area_l=1.0)) == 0.0


def test_C_ma_zero_alpha(cone):
    a = make_aero(alpha=0.0)
    expected = (0.3 * 0.0078 - 0.0006) / (AREF * 0.05)
    assert a.C_ma(cone) == pytest.approx(expected)


def test_C_ma_nonzero_alpha(aero, cone):
    zero_alpha = make_aero(alpha=0.0).C_ma(cone)
    assert aero.C_ma(cone) == pytest.approx(zero_alpha * math.sin(0.05) / 0.05)


def test_pitch_damp_placeholder(aero):
    assert aero.pitch_damp() == 0.0
    assert aero.pitch_damp(SimpleNamespace()) == 0.0


def test_C_m_total_empty_rocket(aero):
    assert aero.C_m_total(SimpleNamespace(parts=[])) == 0.0


def test_C_m_total_cylinder_positive(aero, cylinder):
    assert aero.C_m_total(SimpleNamespace(parts=[cylinder])) > 0


def test_C_m_total_zero_alpha_cylinder(cylinder):
    assert make_aero(alpha=0.0).C_m_total(SimpleNamespace(parts=[cylinder])) == pytest.approx(0.0)


def test_roll_placeholders(aero):
    assert aero.equil_roll() == 0.0
    assert aero.roll_damp() == 0.0


# --------------------------------------------------------------------------
# Centre of pressure
# --------------------------------------------------------------------------

def test_X_lift_uses_cl_when_present(aero):
    assert aero.X_lift(SimpleNamespace(cl=0.42)) == 0.42


def test_X_lift_cylinder_is_half_length(aero, cylinder):
    assert aero.X_lift(cylinder) == pytest.approx(0.5 * cylinder.length)


def test_X_n_missing_area_returns_zero(aero):
    assert aero.X_n(SimpleNamespace()) == 0.0


def test_X_n_equal_areas_returns_zero(aero):
    assert aero.X_n(SimpleNamespace(area_0=1.0, area_l=1.0)) == 0.0


def test_X_n_formula(aero, cone):
    expected = (0.3 * 0.0078 - 0.0006) / 0.0078
    assert aero.X_n(cone) == pytest.approx(expected)


def test_cp_fins_y(aero, fin):
    assert aero.cp_fins(fin)[1] == pytest.approx((0.1 / 3) * 0.4 / 0.3)


def test_cp_fins_subsonic_x(fin):
    x, _ = make_aero(mach=0.5).cp_fins(fin)
    assert x == pytest.approx(0.1 / 3 * 0.4 / 0.3 + 0.09 / 1.8)


def test_cp_fins_supersonic_x(fin):
    x, _ = make_aero(mach=2.0).cp_fins(fin)
    beta = math.sqrt(3.0)
    assert x == pytest.approx((2 * beta - 0.67) / (4 * beta - 1) * 0.15)


def test_cp_fins_transonic_blend_starts_at_subsonic_value(fin):
    x_sub = make_aero(mach=0.5).cp_fins(fin)[0]
    x_at_1 = make_aero(mach=1.0).cp_fins(fin)[0]
    assert x_at_1 == pytest.approx(x_sub)


def test_cp_fins_transonic_blend_ends_at_supersonic_value(fin):
    just_below = make_aero(mach=1.3 - 1e-9).cp_fins(fin)[0]
    at_1_3 = make_aero(mach=1.3).cp_fins(fin)[0]
    assert just_below == pytest.approx(at_1_3, abs=1e-6)


def test_cp_zero_when_no_normal_force(aero):
    assert aero.cp(SimpleNamespace(parts=[])) == 0.0


def test_cp_single_cylinder_is_half_length(aero, cylinder):
    assert aero.cp(SimpleNamespace(parts=[cylinder])) == pytest.approx(0.5)


def test_cp_respects_explicit_lift_centre(aero, cylinder):
    cylinder.cl = 0.4
    assert aero.cp(SimpleNamespace(parts=[cylinder])) == pytest.approx(0.4)


def test_cp_two_cylinders_offsets_second_part(aero, cylinder):
    second = SimpleNamespace(radius=0.05, length=1.0, r=cylinder.r)
    # Equal lift on both parts -> centroid at midpoint of 2 m body
    assert aero.cp(SimpleNamespace(parts=[cylinder, second])) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Drag
# --------------------------------------------------------------------------

def test_skin_friction_mach_zero_known_value():
    a = make_aero(mach=0.0)
    part = SimpleNamespace(length=1.0, radius=0.05)
    expected = 1.48e-2 * 2 * math.pi * 0.05 * 1.0 / AREF
    assert a.skin_friction(SimpleNamespace(parts=[part])) == pytest.approx(expected)


def test_skin_friction_uses_wetted_area_when_given():
    a = make_aero(mach=0.0)
    part = SimpleNamespace(length=1.0, radius=0.05, Awet=0.5)
    assert a.skin_friction(SimpleNamespace(parts=[part])) == pytest.approx(1.48e-2 * 0.5 / AREF)


def test_skin_friction_finite_and_positive(aero, cylinder):
    cd = aero.skin_friction(SimpleNamespace(parts=[cylinder], roughness=0.0))
    assert np.isfinite(cd) and cd > 0


def test_skin_friction_fins_add_drag(aero, cylinder, fin):
    without = aero.skin_friction(SimpleNamespace(parts=[cylinder]))
    with_fins = aero.skin_friction(SimpleNamespace(parts=[cylinder], fins=fin))
    assert with_fins > without


def test_skin_friction_skips_parts_without_length(aero):
    assert aero.skin_friction(SimpleNamespace(parts=[SimpleNamespace(radius=0.05)])) == 0.0


@pytest.mark.parametrize("mach,expected", [
    (0.0, 0.85),
    (0.5, 0.85 * (1 + 0.25 / 4 + 0.0625 / 40)),
    (2.0, 0.85 * (1.84 - 0.76 / 4 + 0.166 / 16 + 0.035 / 64)),
])
def test_cd_blunt_values(mach, expected):
    assert make_aero(mach=mach).cd_blunt() == pytest.approx(expected)


def test_cd_blunt_roughly_continuous_across_mach_1():
    below = make_aero(mach=0.999).cd_blunt()
    above = make_aero(mach=1.001).cd_blunt()
    assert below == pytest.approx(above, abs=0.02)


def test_cd_base_supersonic():
    assert make_aero(mach=2.0).cd_base() == pytest.approx(0.125)


def test_cd_base_subsonic_is_physically_reasonable():
    # NOTE: this currently FAILS because the code has 0.12 + 10.13*M^2.
    # OpenRocket/Barrowman use 0.12 + 0.13*M^2, so 10.13 looks like a typo.
    cd = make_aero(mach=0.5).cd_base()
    assert cd < 1.0


def test_cd_base_roughly_continuous_across_mach_1():
    below = make_aero(mach=0.999).cd_base()
    above = make_aero(mach=1.001).cd_base()
    assert below == pytest.approx(above, abs=0.1)


def test_cd_cone_matches_formula_at_high_mach():
    n = SimpleNamespace(half_angle=0.2)
    s = math.sin(0.2)
    expected = 2.1 * s ** 2 + 0.5 * s / math.sqrt(4 - 1)
    assert make_aero(mach=2.0).cd_cone(n) == pytest.approx(expected)


def test_cd_cone_transonic_fit_hits_endpoints():
    n = SimpleNamespace(half_angle=0.2)
    s = math.sin(0.2)
    at_1 = make_aero(mach=1.0).cd_cone(n)
    assert at_1 == pytest.approx(s)
    below = make_aero(mach=1.3 - 1e-9).cd_cone(n)
    above = make_aero(mach=1.3).cd_cone(n)
    assert below == pytest.approx(above, abs=1e-6)


def test_cd_ogive_supersonic_scales_cone():
    n = SimpleNamespace(shape_factor=1.0, half_angle=0.2, shoulder_angle=0.2)
    a = make_aero(mach=2.0)
    assert a.cd_ogive(n) == pytest.approx((0.72 * 0.25 + 0.82) * a.cd_cone(n))


def test_cd_ogive_mach_zero_equals_shoulder_term():
    n = SimpleNamespace(shape_factor=1.0, half_angle=0.2, shoulder_angle=0.2)
    assert make_aero(mach=0.0).cd_ogive(n) == pytest.approx(0.8 * math.sin(0.2) ** 2)


def test_cd_ogive_subsonic_finite():
    n = SimpleNamespace(shape_factor=1.0, half_angle=0.2, shoulder_angle=0.2)
    assert np.isfinite(make_aero(mach=0.7).cd_ogive(n))


@pytest.fixture
def vk_nose():
    # length = 6*radius -> fn = 3 -> exponent = 1, so result == c3 exactly
    return SimpleNamespace(length=0.6, radius=0.1, cd_data_f3=[[0.5, 1.0, 2.0], [0.2, 0.3, 0.25]])


def test_cd_vk_interpolates_inside_data_range(vk_nose):
    assert make_aero(mach=1.0).cd_vk(vk_nose) == pytest.approx(0.3)


def test_cd_vk_clamps_below_range(vk_nose):
    assert make_aero(mach=0.1).cd_vk(vk_nose) == pytest.approx(0.2)


def test_cd_vk_clamps_above_range(vk_nose):
    assert make_aero(mach=3.0).cd_vk(vk_nose) == pytest.approx(0.25)


def test_cd_boattail_missing_geometry_returns_zero(aero):
    assert aero.cd_boattail(SimpleNamespace()) == 0.0


def test_cd_boattail_no_taper_returns_zero(aero):
    bt = SimpleNamespace(length=0.1, radius_0=0.05, radius_l=0.05, area_0=0.01, area_l=0.01)
    assert aero.cd_boattail(bt) == 0.0


@pytest.mark.parametrize("length,coeff", [
    (0.05, 1.0),   # L/dr*0.5 = 0.5 -> full base drag
    (0.20, 0.5),   # ratio 2 -> half
    (0.30, 0.0),   # ratio 3 -> none
])
def test_cd_boattail_coefficient_bands(length, coeff):
    a = make_aero(mach=2.0)  # supersonic: avoids the subsonic cd_base issue
    bt = SimpleNamespace(length=length, radius_0=0.10, radius_l=0.05, area_0=0.0314, area_l=0.0078)
    assert a.cd_boattail(bt) == pytest.approx(coeff * a.cd_base())


def test_cd_fins_zero_at_mach_zero(fin):
    assert make_aero(mach=0.0).cd_fins(fin) == pytest.approx(0.0)


def test_cd_fins_supersonic_unswept():
    f = SimpleNamespace(sweep_length=0.0, span=0.1)
    expected = 1.214 - 0.502 / 4 + 0.1085 / 16
    assert make_aero(mach=2.0).cd_fins(f) == pytest.approx(expected)


def test_cd_fins_sweep_reduces_drag():
    unswept = SimpleNamespace(sweep_length=0.0, span=0.1)
    swept = SimpleNamespace(sweep_length=0.1, span=0.1)
    a = make_aero(mach=2.0)
    assert a.cd_fins(swept) < a.cd_fins(unswept)


# --------------------------------------------------------------------------
# Total drag
# --------------------------------------------------------------------------

@pytest.fixture
def simple_rocket():
    nose = nosecone_vk(length=0.6, radius=0.1, area_0=0.0, area_l=0.0314,
                       cd_data_f3=[[0.5, 1.0, 2.0], [0.2, 0.3, 0.25]])
    tube = body_tube(length=1.0, radius=0.1, area_0=0.0314, area_l=0.0314)
    return SimpleNamespace(parts=[nose, tube], roughness=0.0)


def test_cd_total_empty_rocket(aero):
    assert aero.cd_total(SimpleNamespace(parts=[])) == 0.0


def test_cd_total_sums_components_at_zero_alpha(simple_rocket):
    a = make_aero(mach=0.5, alpha=0.0)
    nose, tube = simple_rocket.parts
    expected = (
        a.cd_vk(nose) * nose.area_l / AREF
        + a.cd_base() * tube.area_l / AREF
        + a.skin_friction(simple_rocket)
    )
    assert a.cd_total(simple_rocket) == pytest.approx(expected)


def test_cd_total_uses_boattail_drag_when_aft_part_is_boattail(simple_rocket):
    bt = boattail(length=0.05, radius_0=0.1, radius_l=0.05, area_0=0.0314, area_l=0.0078)
    simple_rocket.parts.append(bt)
    a = make_aero(mach=2.0, alpha=0.0)
    nose = simple_rocket.parts[0]
    expected = (
        a.cd_vk(nose) * nose.area_l / AREF
        + a.cd_boattail(bt) * bt.area_l / AREF
        + a.skin_friction(simple_rocket)
    )
    assert a.cd_total(simple_rocket) == pytest.approx(expected)


def test_cd_total_fins_add_drag(simple_rocket, fin):
    a = make_aero(mach=2.0, alpha=0.0)
    without = a.cd_total(simple_rocket)
    simple_rocket.fins = fin
    assert a.cd_total(simple_rocket) > without


def test_cd_total_angle_of_attack_scale_factor(simple_rocket):
    a0 = make_aero(mach=0.5, alpha=0.0).cd_total(simple_rocket)
    a10 = make_aero(mach=0.5, alpha=math.radians(10)).cd_total(simple_rocket)
    scale = -(3 / 24565) * 10 ** 3 + (9 / 2890) * 10 ** 2 + 1.0
    assert a10 / a0 == pytest.approx(scale)


def test_cd_total_alpha_scale_continuous_at_17_degrees(simple_rocket):
    below = make_aero(mach=0.5, alpha=math.radians(17.0)).cd_total(simple_rocket)
    above = make_aero(mach=0.5, alpha=math.radians(17.0001)).cd_total(simple_rocket)
    assert below == pytest.approx(above, rel=1e-3)


@pytest.mark.parametrize("mach", [0.1, 0.5, 0.95, 1.0, 1.2, 2.0, 3.0])
def test_cd_total_finite_across_mach_range(simple_rocket, mach):
    cd = make_aero(mach=mach, alpha=0.0).cd_total(simple_rocket)
    assert np.isfinite(cd)
