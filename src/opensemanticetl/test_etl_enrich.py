#!/usr/bin/python3
# -*- coding: utf-8 -*-

import contextlib
import io
import threading
import unittest

import etl_enrich


class TestStratifiedSampler(unittest.TestCase):

    def test_invalid_budget(self):

        with self.assertRaises(ValueError):
            etl_enrich.StratifiedSampler(budget=0)

    def test_reserve_one_slot_per_stratum(self):

        sampler = etl_enrich.StratifiedSampler(budget=3, seed=0)

        for i in range(10):
            sampler.add('a', 'doc-a-{}'.format(i))
            sampler.add('b', 'doc-b-{}'.format(i))
            sampler.add('c', 'doc-c-{}'.format(i))

        allocation = sampler.allocation()

        # every stratum gets at least its reserved slot
        self.assertEqual(set(allocation), {'a', 'b', 'c'})
        self.assertEqual(sum(allocation.values()), 3)

    def test_budget_smaller_than_strata_raises(self):

        sampler = etl_enrich.StratifiedSampler(budget=2, seed=0)

        sampler.add('a', 'doc-1')
        sampler.add('b', 'doc-2')
        sampler.add('c', 'doc-3')

        with self.assertRaises(ValueError):
            sampler.allocation()

        with self.assertRaises(ValueError):
            sampler.sample()

    def test_proportional_to_remaining_capacity_with_largest_remainder(self):

        # stratum sizes 8, 3, 2 and budget 6:
        # 1 reserved slot per stratum, remaining budget 3 distributed
        # proportional to remaining capacities 7, 2, 1 (sum 10):
        # shares 2.1, 0.6, 0.3 -> floors 2, 0, 0,
        # leftover 1 to largest remainder (stratum b)
        sampler = etl_enrich.StratifiedSampler(budget=6, seed=0)

        for i in range(8):
            sampler.add('a', 'doc-a-{}'.format(i))
        for i in range(3):
            sampler.add('b', 'doc-b-{}'.format(i))
        for i in range(2):
            sampler.add('c', 'doc-c-{}'.format(i))

        allocation = sampler.allocation()

        self.assertEqual(allocation, {'a': 3, 'b': 2, 'c': 1})

    def test_remainder_tie_broken_by_stratum_key(self):

        # equal strata, equal remainders: the leftover slot
        # goes to the first stratum in stable key order
        sampler = etl_enrich.StratifiedSampler(budget=3, seed=0)

        for i in range(5):
            sampler.add('a', 'doc-a-{}'.format(i))
            sampler.add('b', 'doc-b-{}'.format(i))

        allocation = sampler.allocation()

        self.assertEqual(allocation, {'a': 2, 'b': 1})

    def test_budget_covers_all_documents(self):

        sampler = etl_enrich.StratifiedSampler(budget=10, seed=0)

        for i in range(3):
            sampler.add('a', 'doc-a-{}'.format(i))
        for i in range(2):
            sampler.add('b', 'doc-b-{}'.format(i))

        sample = sampler.sample()

        self.assertEqual(sample, [
            ('a', 3, ['doc-a-0', 'doc-a-1', 'doc-a-2']),
            ('b', 2, ['doc-b-0', 'doc-b-1']),
        ])

    def test_missing_value_is_own_stratum(self):

        sampler = etl_enrich.StratifiedSampler(budget=2, seed=0)

        sampler.add('x', 'doc-1')
        sampler.add(None, 'doc-2')

        sample = sampler.sample()

        # documents without value form an own stratum, ordered last
        self.assertEqual([key for key, size, docids in sample], ['x', None])
        self.assertEqual(sample[0], ('x', 1, ['doc-1']))
        self.assertEqual(sample[1], (None, 1, ['doc-2']))

    def test_sample_stable_regardless_of_order(self):

        docids = ['doc-{}'.format(i) for i in range(200)]
        strata = ['a', 'b', 'c']

        samples = []
        for order in (docids, list(reversed(docids)), docids[100:] + docids[:100]):
            sampler = etl_enrich.StratifiedSampler(budget=30, seed=7)
            for docid in order:
                sampler.add(strata[hash(docid) % 3], docid)
            samples.append(sampler.sample())

        self.assertEqual(samples[0], samples[1])
        self.assertEqual(samples[0], samples[2])

    def test_sample_depends_on_seed(self):

        docids = ['doc-{}'.format(i) for i in range(200)]

        samples = []
        for seed in (1, 2):
            sampler = etl_enrich.StratifiedSampler(budget=20, seed=seed)
            for docid in docids:
                sampler.add('a', docid)
            samples.append(sampler.sample()[0][2])

        self.assertNotEqual(samples[0], samples[1])

    def test_memory_bounded_by_budget(self):

        # single stratum: at most budget ids are kept
        sampler = etl_enrich.StratifiedSampler(budget=10, seed=0)
        for i in range(10000):
            sampler.add('a', 'doc-{}'.format(i))

        self.assertEqual(len(sampler.strata['a']['heap']), 10)

        # every additional stratum reserves a slot and shrinks the heaps
        sampler = etl_enrich.StratifiedSampler(budget=10, seed=0)
        for i in range(1000):
            sampler.add('a', 'doc-a-{}'.format(i))
        for i in range(1000):
            sampler.add('b', 'doc-b-{}'.format(i))

        self.assertEqual(len(sampler.strata['a']['heap']), 9)
        self.assertEqual(len(sampler.strata['b']['heap']), 9)

    def test_allocation_invariants(self):

        # for many stratum sizes and budgets: every stratum gets its
        # reserved slot, no stratum more than its size, and in sum
        # exactly the budget (or all documents, if fewer exist)
        for sizes in ([1], [1, 1, 1], [5, 5], [8, 3, 2], [100, 1, 1, 50]):

            total = sum(sizes)

            for budget in range(1, total + 3):

                sampler = etl_enrich.StratifiedSampler(budget=budget, seed=0)
                for stratum, size in enumerate(sizes):
                    for i in range(size):
                        sampler.add(stratum, 'doc-{}-{}'.format(stratum, i))

                if budget < len(sizes):
                    with self.assertRaises(ValueError):
                        sampler.allocation()
                    continue

                allocation = sampler.allocation()

                for stratum, size in enumerate(sizes):
                    self.assertGreaterEqual(allocation[stratum], 1)
                    self.assertLessEqual(allocation[stratum], size)

                self.assertEqual(sum(allocation.values()),
                                 min(budget, total))


class FakeResults(object):

    def __init__(self, docs):
        self.docs = docs

    def __len__(self):
        return len(self.docs)

    def __iter__(self):
        return iter(self.docs)


class FakeSolr(object):
    """Emulates the search behaviour enrich_query relies on.

    Cursor scans (sampling) iterate all documents in the given order.
    The classic paging loop gets the not yet enriched documents
    (like the real query, which excludes documents marked as enriched).
    """

    def __init__(self, docs):
        self.docs = list(docs)
        self.enriched = set()
        self.searches = []

    def search(self, q, **kwargs):

        self.searches.append(kwargs)

        if 'cursorMark' in kwargs:
            return FakeResults(list(self.docs))

        rows = kwargs.get('rows', 10)
        remaining = [doc for doc in self.docs if doc['id'] not in self.enriched]

        return FakeResults(remaining[:rows])


def build_etl(docs, rows_per_step=100, threads_max=1):
    """ETL_Enrich instance without config files and without a real Solr."""

    etl = etl_enrich.ETL_Enrich.__new__(etl_enrich.ETL_Enrich)

    etl.verbose = False
    etl.config = {'plugins': []}
    etl.solr = FakeSolr(docs)
    etl.rows_per_step = rows_per_step
    etl.threads_max = threads_max

    etl.work_in_progress = []
    etl.delete_from_work_in_progress_after_commit = []
    etl.work_in_progress_lock = threading.Lock()
    etl.delete_from_work_in_progress_lock = threading.Lock()
    etl.e_job_done = threading.Event()

    etl.enriched = []

    def fake_enrich_document(docid):
        etl.enriched.append(docid)
        etl.solr.enriched.add(docid)
        # mirror the bookkeeping of the real enrich_document
        etl.work_in_progress_lock.acquire()
        if docid in etl.work_in_progress:
            etl.delete_from_work_in_progress_after_commit.append(docid)
        etl.work_in_progress_lock.release()
        etl.e_job_done.set()

    etl.enrich_document = fake_enrich_document
    etl.commit = lambda: None

    return etl


def enrich_query(etl, *args, **kwargs):
    """Run enrich_query with captured stdout, return the output."""

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        etl.enrich_query(*args, **kwargs)

    return stdout.getvalue()


class TestEnrichQuery(unittest.TestCase):

    docs = []
    for i in range(50):
        docs.append({'id': 'pdf-{}'.format(i),
                     'content_type_s': 'application/pdf'})
    for i in range(5):
        docs.append({'id': 'mail-{}'.format(i),
                     'content_type_s': 'message/rfc822'})
    for i in range(3):
        # documents without a value in the stratification field
        docs.append({'id': 'none-{}'.format(i)})

    def expected_sample(self, sample_size, sample_field, sample_seed):

        sampler = etl_enrich.StratifiedSampler(
            budget=sample_size, seed=sample_seed)
        for doc in self.docs:
            sampler.add(doc.get(sample_field, None), doc['id'])

        return sampler.sample()

    def test_enrich_query_without_sampling_enriches_all(self):

        etl = build_etl(self.docs, rows_per_step=10)

        enrich_query(etl, '*:*')

        self.assertEqual(sorted(etl.enriched),
                         sorted(doc['id'] for doc in self.docs))

    def test_enrich_query_without_sampling_threaded(self):

        etl = build_etl(self.docs, rows_per_step=10, threads_max=2)

        enrich_query(etl, '*:*')

        self.assertEqual(sorted(etl.enriched),
                         sorted(doc['id'] for doc in self.docs))

    def test_enrich_query_sample_enriches_only_sample(self):

        etl = build_etl(self.docs, rows_per_step=7)

        output = enrich_query(etl, '*:*', sample_size=13,
                              sample_field='content_type_s', sample_seed=42)

        expected = {}
        for key, stratum_size, docids in self.expected_sample(
                13, 'content_type_s', 42):
            expected[key] = (stratum_size, docids)

        # only the sampled documents were enriched
        enriched = sorted(etl.enriched)
        self.assertEqual(enriched, sorted(
            docid for key, (size, docids) in expected.items()
            for docid in docids))
        self.assertEqual(len(enriched), 13)

        # every stratum got its reserved slot, i.e. rare strata are sampled, too
        self.assertTrue(any(docid.startswith('mail-') for docid in enriched))
        self.assertTrue(any(docid.startswith('none-') for docid in enriched))

        # total and sampled count reported per stratum
        for key, (stratum_size, docids) in expected.items():
            if key is None:
                key = '(missing)'
            self.assertIn("Stratum {}: sampled {} of {} documents".format(
                key, len(docids), stratum_size), output)

        self.assertIn("Enriched 13 documents", output)

    def test_enrich_query_sample_scans_with_cursor(self):

        etl = build_etl(self.docs, rows_per_step=7)

        enrich_query(etl, '*:*', sample_size=13,
                     sample_field='content_type_s', sample_seed=42)

        scan = etl.solr.searches[0]
        self.assertEqual(scan['cursorMark'], '*')
        self.assertEqual(scan['sort'], 'id asc')
        self.assertEqual(scan['fl'], 'id,content_type_s')

    def test_enrich_query_sample_independent_of_page_size_and_order(self):

        samples = []

        for rows_per_step, docs in ((3, self.docs),
                                    (100, list(reversed(self.docs)))):
            etl = build_etl(docs, rows_per_step=rows_per_step)
            enrich_query(etl, '*:*', sample_size=13,
                         sample_field='content_type_s', sample_seed=42)
            samples.append(sorted(etl.enriched))

        self.assertEqual(samples[0], samples[1])

    def test_enrich_query_sample_budget_smaller_than_strata_raises(self):

        etl = build_etl(self.docs)

        with self.assertRaises(ValueError):
            enrich_query(etl, '*:*', sample_size=2,
                         sample_field='content_type_s')

        # nothing was enriched
        self.assertEqual(etl.enriched, [])

    def test_enrich_query_sample_needs_field(self):

        etl = build_etl(self.docs)

        with self.assertRaises(ValueError):
            enrich_query(etl, '*:*', sample_size=10)

        self.assertEqual(etl.enriched, [])

    def test_enrich_query_sample_field_needs_size(self):

        etl = build_etl(self.docs)

        with self.assertRaises(ValueError):
            enrich_query(etl, '*:*', sample_field='content_type_s')

        self.assertEqual(etl.enriched, [])


if __name__ == '__main__':
    unittest.main()
