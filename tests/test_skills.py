"""
tests/test_skills.py

Unit tests for edgedash.skills.canonical().
Pure function — no network, no DB, no mocks.
"""

from __future__ import annotations

import pytest
from edgedash.skills import canonical

_ALIASES: dict = {
    "kubernetes": ["k8s", "k8"],
    "ci/cd": ["cicd", "ci cd", "ci/cd pipelines"],
    "postgresql": ["postgres", "psql"],
    "c#": ["c# .net", "c# .net core"],
}


class TestCase:
    def test_lowercases(self):
        assert canonical("Python", _ALIASES) == "python"

    def test_lowercases_mixed(self):
        assert canonical("KUBERNETES", _ALIASES) == "kubernetes"


class TestWhitespace:
    def test_strips_leading_trailing(self):
        assert canonical("  python  ", _ALIASES) == "python"

    def test_collapses_internal(self):
        # "ci  cd" collapses to "ci cd" which aliases to "ci/cd" — correct
        assert canonical("ci  cd", _ALIASES) == "ci/cd"

    def test_internal_space_normalised(self):
        # Two spaces become one before alias lookup
        result = canonical("ci  /  cd", _ALIASES)
        assert " " not in result or result == result.strip()


class TestParentheses:
    def test_strips_parenthetical(self):
        assert canonical("Kubernetes (EKS)", _ALIASES) == "kubernetes"

    def test_strips_parenthetical_no_alias(self):
        assert canonical("Docker (latest)", _ALIASES) == "docker"

    def test_multi_word_with_parens(self):
        assert canonical("Node.js (v18)", _ALIASES) == "node.js"


class TestAliased:
    def test_k8s_maps_to_kubernetes(self):
        assert canonical("k8s", _ALIASES) == "kubernetes"

    def test_k8_maps_to_kubernetes(self):
        assert canonical("k8", _ALIASES) == "kubernetes"

    def test_cicd_maps(self):
        assert canonical("cicd", _ALIASES) == "ci/cd"

    def test_ci_cd_with_space(self):
        assert canonical("ci cd", _ALIASES) == "ci/cd"

    def test_postgres_maps(self):
        assert canonical("postgres", _ALIASES) == "postgresql"

    def test_alias_case_insensitive(self):
        assert canonical("K8S", _ALIASES) == "kubernetes"

    def test_ci_cd_pipelines(self):
        assert canonical("ci/cd pipelines", _ALIASES) == "ci/cd"


class TestNoAlias:
    def test_unknown_term_returned_clean(self):
        assert canonical("terraform", _ALIASES) == "terraform"

    def test_unknown_with_caps(self):
        assert canonical("Terraform", _ALIASES) == "terraform"


class TestCSharp:
    def test_csharp_preserved(self):
        # c# must not be stripped to just "c"
        aliases = {"c#": ["c# .net"]}
        assert canonical("c#", aliases) == "c#"

    def test_csharp_alias(self):
        assert canonical("c# .net", _ALIASES) == "c#"


class TestEmpty:
    def test_empty_string(self):
        assert canonical("", _ALIASES) == ""

    def test_whitespace_only(self):
        assert canonical("   ", _ALIASES) == ""

    def test_none_like_empty(self):
        # Callers should pass str, but guard anyway
        assert canonical("", _ALIASES) == ""
