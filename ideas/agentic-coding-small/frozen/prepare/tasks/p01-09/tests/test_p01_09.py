from dataclasses import FrozenInstanceError
import pytest
from semver import Version
from version_ranges import VersionRange

def test_version_parse_fields_and_roundtrip():
    v = Version.parse('12.3.0-rc.2.a-b+build.001')
    assert (v.major, v.minor, v.patch) == (12, 3, 0)
    assert v.prerelease == ('rc', '2', 'a-b')
    assert v.build == ('build', '001')
    assert str(v) == '12.3.0-rc.2.a-b+build.001'
    assert str(Version(0, 0, 0)) == '0.0.0'
    assert Version.parse('1.2.3').prerelease == ()
    assert Version.parse('1.2.3').build == ()
    with pytest.raises(FrozenInstanceError):
        v.major = 1

def test_reject_invalid_versions():
    for text in ('1.2', 'v1.2.3', ' 1.2.3', '1.2.3\n', '01.2.3', '1.02.3', '1.2.03', '1.2.3-', '1.2.3+', '1.2.3-a..b', '1.2.3-01', '1.2.3-a.00', '1.2.3+a_b', '1.2.3+abc.', '１.2.3'):
        with pytest.raises(ValueError):
            Version.parse(text)

def test_precedence_ordering_and_all_operators():
    texts = ['1.0.0-alpha', '1.0.0-alpha.1', '1.0.0-alpha.2', '1.0.0-alpha.10', '1.0.0-alpha.beta', '1.0.0-beta', '1.0.0-beta.2', '1.0.0-beta.11', '1.0.0-rc.1', '1.0.0', '1.0.1', '1.1.0', '2.0.0']
    versions = [Version.parse(t) for t in texts]
    assert sorted(reversed(versions)) == versions
    for a, b in zip(versions, versions[1:]):
        assert a < b and a <= b and b > a and b >= a and a != b
    assert Version.parse('1.0.0-9') < Version.parse('1.0.0-A') < Version.parse('1.0.0-a')

def test_build_metadata_ignored_for_equality_and_hash():
    a = Version.parse('1.2.3-rc.1+abc')
    b = Version.parse('1.2.3-rc.1+xyz.001')
    assert a == b and a <= b and a >= b
    assert not (a < b or a > b or a != b)
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert VersionRange('=1.2.3+first').contains('1.2.3+second') is True

def test_comparator_conjunctions_and_alternatives():
    r = VersionRange(' >=1.2.0 <2.0.0 || >3.0.0 <=3.2.0 ')
    for version in ('1.2.0', '1.9.9', '2.0.0-alpha', '3.0.1', '3.2.0'):
        assert r.contains(Version.parse(version)) is True
    for version in ('1.1.9', '2.0.0', '3.0.0', '3.2.1'):
        assert r.contains(version) is False
    assert VersionRange('1.2.3').contains('1.2.3') is True
    assert VersionRange('1.2.3').contains('1.2.4') is False
    assert VersionRange('>=2.0.0 <1.0.0').contains('1.5.0') is False

def test_wildcard_ranges_and_prerelease_boundaries():
    assert VersionRange('*').contains('0.0.0-alpha') is True
    for spec, yes, no in [('1.*', ['1.0.0', '1.9.9', '2.0.0-alpha'], ['1.0.0-alpha', '2.0.0']), ('1.2.*', ['1.2.0', '1.2.9', '1.3.0-beta'], ['1.1.9', '1.3.0'])]:
        r = VersionRange(spec)
        assert all(r.contains(v) for v in yes)
        assert all(not r.contains(v) for v in no)

def test_caret_and_tilde_expansion():
    for spec, low, inside, upper in [('^1.2.3', '1.2.3', '1.9.9', '2.0.0'), ('^0.2.3', '0.2.3', '0.2.9', '0.3.0'), ('^0.0.3', '0.0.3', '0.0.4-alpha', '0.0.4'), ('^0.0.0', '0.0.0', '0.0.1-alpha', '0.0.1'), ('~1.2.3', '1.2.3', '1.2.9', '1.3.0')]:
        r = VersionRange(spec)
        assert r.contains(low) and r.contains(inside)
        assert not r.contains(upper)
        assert not r.contains('0.0.0-alpha')
    assert VersionRange('^1.2.3-beta.1').contains('1.2.3-beta.2') is True
    assert VersionRange('~1.2.3').contains('1.2.2') is False

def test_invalid_range_syntax_and_version_argument():
    for spec in ('', ' ', '||1.2.3', '1.2.3||', '* || bad', '* bad', '>= 1.2.3', '1.2', '1', '1.x', '01.*', '1.02.*', '>=1.*', '1.2.3.*', '!=1.2.3', '1.2.3,2.0.0', '1.0.0 - 2.0.0', '^1.2', '~1', '1.2.3 | 2.0.0'):
        with pytest.raises(ValueError):
            VersionRange(spec)
    with pytest.raises(ValueError):
        VersionRange('*').contains('bad')
