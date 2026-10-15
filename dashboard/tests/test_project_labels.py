"""Tests for dashboard.project_labels — one label per configured project root."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from dashboard.project_labels import ProjectLabels, basename_label

_SAME_BASENAME = ('/srv/team-a/app', '/srv/team-b/app', '/srv/other')
_DEEP_COLLISION = ('/x/a/proj', '/y/a/proj', '/b/proj')
_SUFFIX_OF_ANOTHER = ('/a/proj', '/b/a/proj')
_MIXED = ('/', '/a/proj', '/b/a/proj', '/c/b/a/proj', '/srv/alpha')


def _labels_by_root(roots) -> dict[str, str]:
    labels = ProjectLabels.of_roots(roots)
    return {str(root): labels.of(root) for root in roots}


class TestOfRoots:
    def test_distinct_basenames_stay_bare(self):
        labels = ProjectLabels.of_roots(['/srv/alpha', '/srv/beta'])

        assert labels.of('/srv/alpha') == 'alpha'
        assert labels.of('/srv/beta') == 'beta'

    def test_shared_basenames_get_their_shortest_unique_suffix(self):
        assert _labels_by_root(_SAME_BASENAME) == {
            '/srv/team-a/app': 'team-a/app',
            '/srv/team-b/app': 'team-b/app',
            '/srv/other': 'other',
        }

    def test_suffix_length_is_chosen_per_root(self):
        assert _labels_by_root(_DEEP_COLLISION) == {
            '/x/a/proj': 'x/a/proj',
            '/y/a/proj': 'y/a/proj',
            '/b/proj': 'b/proj',
        }

    def test_a_root_that_is_a_suffix_of_another_is_labelled_by_its_full_path(self):
        assert _labels_by_root(_SUFFIX_OF_ANOTHER) == {
            '/a/proj': '/a/proj',
            '/b/a/proj': 'b/a/proj',
        }

    def test_the_basename_less_root_is_labelled_by_its_full_path(self):
        assert ProjectLabels.of_roots(['/']).of('/') == '/'

    def test_trailing_slash_and_path_inputs_agree(self):
        labels = ProjectLabels.of_roots(['/srv/team-a/app', '/srv/team-b/app'])

        assert labels.of('/srv/team-a/app/') == labels.of(Path('/srv/team-a/app')) == 'team-a/app'

    @pytest.mark.parametrize('roots', [_SAME_BASENAME, _DEEP_COLLISION, _SUFFIX_OF_ANOTHER, _MIXED])
    def test_labels_do_not_depend_on_root_order(self, roots):
        forward = ProjectLabels.of_roots(roots)
        backward = ProjectLabels.of_roots(reversed(roots))

        assert {root: backward.of(root) for root in roots} == {root: forward.of(root) for root in roots}

    def test_a_repeated_root_is_configured_once_and_keeps_its_basename(self):
        labels = ProjectLabels.of_roots(['/srv/alpha', '/srv/alpha'])

        assert labels.roots == (Path('/srv/alpha'),)
        assert labels.of('/srv/alpha') == 'alpha'

    def test_roots_preserve_first_seen_order(self):
        labels = ProjectLabels.of_roots(['/srv/beta', Path('/srv/alpha'), '/srv/beta/', '/srv/gamma'])

        assert labels.roots == (Path('/srv/beta'), Path('/srv/alpha'), Path('/srv/gamma'))

    def test_an_unconfigured_root_falls_back_to_its_basename(self):
        labels = ProjectLabels.of_roots(['/srv/team-a/app', '/srv/team-b/app'])

        assert labels.of('/elsewhere/app') == 'app'

    @pytest.mark.parametrize(
        'roots', [_SAME_BASENAME, _DEEP_COLLISION, _SUFFIX_OF_ANOTHER, _MIXED, ('/srv/alpha', '/srv/beta')]
    )
    def test_labels_are_pairwise_distinct(self, roots):
        labels = ProjectLabels.of_roots(roots)

        for left, right in itertools.combinations(labels.roots, 2):
            assert labels.of(left) != labels.of(right), (left, right)


class TestRootFor:
    @pytest.mark.parametrize('roots', [_SAME_BASENAME, _DEEP_COLLISION, _SUFFIX_OF_ANOTHER, _MIXED])
    def test_every_configured_root_round_trips(self, roots):
        labels = ProjectLabels.of_roots(roots)

        for root in labels.roots:
            assert labels.root_for(labels.of(root)) == root

    def test_a_basename_shared_by_two_configured_roots_names_neither(self):
        labels = ProjectLabels.of_roots(['/srv/team-a/app', '/srv/team-b/app'])

        assert labels.root_for('app') is None

    def test_an_unknown_label_names_no_root(self):
        labels = ProjectLabels.of_roots(['/srv/team-a/app', '/srv/team-b/app'])

        assert labels.root_for('nope') is None


class TestBasenameLabel:
    def test_trailing_slash_is_stripped(self):
        assert basename_label('/a/b/') == 'b'

    def test_a_root_without_a_basename_falls_back_to_the_full_string(self):
        assert basename_label('/') == '/'

    def test_path_and_str_agree(self):
        assert basename_label(Path('/a/b')) == basename_label('/a/b') == 'b'

    def test_composes_the_project_basename(self):
        assert basename_label('/home/leo/src/dark-factory') == 'dark-factory'
