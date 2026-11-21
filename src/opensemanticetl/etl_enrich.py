#!/usr/bin/python3
# -*- coding: utf-8 -*-

from etl import ETL
import pysolr
import export_solr
import hashlib
import heapq
import importlib
import threading

# Todo: Abstraction of querying data to a function of output plugin
# so this will work for other index or database than Solr, too

# Todo: Yet problem, if you run only enrichment plugins (i.e. OCR) without container plugins (i.e. mailbox extractor or ZIP archive extractor), if you want to run this plugin with a container plugin:
# because the container is marked as done in first run (but was not extract because container/extraction plugin not active) so next run with this continer_plugins/extractors wont enrich them anymore

# Since we use enrichment of queries only for OCR after indexing in Open Semantic Desktop Search where we know we call the OCR plugin with all container plugins, fixing or maybe making this perfect later (maybe better classification/management of container plugins or defining in plugins, if they need access to file)


#
# Stratified sampling of the documents matching a query
#
# For spot checking a limited count of documents (i.e. before running an
# enrichment on the full index), stratified by a single valued field,
# so that rare values (long tail) are represented in the sample, too.
#

class StratifiedSampler(object):

    # stratum key for documents without a value in the stratification field
    MISSING = None

    def __init__(self, budget, seed=0):

        if budget < 1:
            raise ValueError(
                "Sample size (budget) has to be at least 1, but is {}".format(budget))

        self.budget = budget
        self.seed = seed

        # stratum key -> {'count': how many documents seen, 'heap': best ranked ids}
        self.strata = {}

    def rank(self, docid):

        # stable rank of a document inside its stratum:
        # since it depends only on seed and document id, neither the page size
        # nor the order in which the index returns documents changes the sample

        hasher = hashlib.sha256()
        hasher.update(str(self.seed).encode('utf-8'))
        hasher.update(b'\x00')
        hasher.update(str(docid).encode('utf-8'))

        return int.from_bytes(hasher.digest()[:16], 'big')

    def max_per_stratum(self):

        # since every stratum gets at least its one reserved slot,
        # no stratum can be allotted more than budget - (strata - 1) samples

        return self.budget - len(self.strata) + 1

    def add(self, key, docid):

        if key not in self.strata:

            self.strata[key] = {'count': 0, 'heap': []}

            # one more stratum reserves one more slot of the budget,
            # so the maximal possible allotment per stratum shrinks:
            # trim the heaps to the new maximum
            self.trim_heaps()

        stratum = self.strata[key]
        stratum['count'] += 1

        max_per_stratum = self.max_per_stratum()

        if max_per_stratum < 1:
            # more strata than budget: sampling not possible
            # (will raise an error in allocation), but keep counting
            return

        # keep only the best ranked ids per stratum (heap with worst kept id on top),
        # so memory is bounded by the budget, not by the count of matching documents
        entry = (-self.rank(docid), docid)

        heap = stratum['heap']

        if len(heap) < max_per_stratum:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    def trim_heaps(self):

        max_per_stratum = max(self.max_per_stratum(), 0)

        for stratum in self.strata.values():
            heap = stratum['heap']
            while len(heap) > max_per_stratum:
                heapq.heappop(heap)

    @staticmethod
    def sort_key(key):

        # stable order of strata by key, stratum of documents without value last
        return (key is None, str(key))

    def allocation(self):

        # how many samples each stratum is allotted:
        # one reserved slot per stratum, remaining budget distributed
        # proportional to the remaining capacities (stratum size - 1)

        strata_count = len(self.strata)

        if self.budget < strata_count:
            raise ValueError(
                "Sample size (budget) {} is smaller than the count of strata {}".format(
                    self.budget, strata_count))

        remaining_budget = self.budget - strata_count

        allocation = {}
        capacities = {}
        for key, stratum in self.strata.items():
            # reserved slot for every stratum
            allocation[key] = 1
            # remaining capacity after the reserved slot
            capacities[key] = stratum['count'] - 1

        remaining_capacity = sum(capacities.values())

        if remaining_budget > 0 and remaining_capacity > 0:

            if remaining_budget >= remaining_capacity:

                # budget covers all remaining candidates: allot every document
                for key in allocation:
                    allocation[key] += capacities[key]

            else:

                # distribute proportional to remaining capacity,
                # rounding down first (exact integer math, same denominator
                # for all strata, so remainders are comparable as integers)
                remainders = {}
                distributed = 0

                for key, capacity in capacities.items():
                    share = remaining_budget * capacity
                    floor = share // remaining_capacity
                    allocation[key] += floor
                    distributed += floor
                    remainders[key] = share % remaining_capacity

                # hand out the rest by largest remainder,
                # ties broken by stable stratum key order
                leftover = remaining_budget - distributed

                by_remainder = sorted(self.strata,
                                      key=lambda key: (-remainders[key], self.sort_key(key)))

                for key in by_remainder[:leftover]:
                    allocation[key] += 1

        return allocation

    def sample(self):

        # the stratified sample as list of
        # (stratum key, stratum size, [sampled document ids])
        # in stable stratum order

        allocation = self.allocation()

        sample = []

        for key in sorted(self.strata, key=self.sort_key):

            stratum = self.strata[key]

            best = heapq.nsmallest(allocation[key], stratum['heap'])

            docids = sorted(docid for (negrank, docid) in best)

            sample.append((key, stratum['count'], docids))

        return sample


class ETL_Enrich(ETL):

    def __init__(self, plugins=(), verbose=False):

        ETL.__init__(self, plugins=list(plugins), verbose=verbose)

        self.read_configfile('/etc/etl/config')
        self.read_configfile('/etc/opensemanticsearch/etl')
        self.read_configfile('/etc/opensemanticsearch/enhancer-rdf')

        self.fields = self.getfieldnames_from_plugins()

        # init exporter	(todo: exporter as extended PySolr)
        self.export_solr = export_solr.export_solr()

        # init PySolr
        solr_uri = self.config['solr']
        if not solr_uri.endswith('/'):
            solr_uri += '/'
        solr_uri += self.config['index']

        self.solr = pysolr.Solr(solr_uri)

        self.threads_max = None

        # if not set explicit, autodetection of count of CPUs for amount of threads
        if not self.threads_max:
            import multiprocessing
            self.threads_max = multiprocessing.cpu_count()
            if self.verbose:
                print("Setting threads to count of CPUs: " +
                      str(self.threads_max))

        self.rows_per_step = 100
        if self.rows_per_step < self.threads_max * 2:
            self.rows_per_step = self.threads_max * 2

        self.work_in_progress = []
        self.delete_from_work_in_progress_lock = threading.Lock()

        self.delete_from_work_in_progress_after_commit = []
        self.work_in_progress_lock = threading.Lock()

        self.e_job_done = threading.Event()

    #
    # get all the fields needed by all plugins for analysis
    #

    def getfieldnames_from_plugins(self):

        # the field id is needed for every enrichment
        fields = ['id']

        # read all fieldnames, the plugins need to analyze
        for plugin in self.config['plugins']:

            module = importlib.import_module(plugin)
            objectreference = getattr(module, plugin, False)
            if objectreference:
                modulefields = getattr(objectreference, 'fields', False)
            if modulefields:
                for field in modulefields:
                    # only if not there yet from other plugin
                    if not field in fields:
                        fields.append(field)

        return fields

    #
    # Start ETL process / run of all set plugins
    #

    def enrich_document(self, docid):

        try:

            if self.verbose:
                print("Enriching {}".format((docid)))

            parameters = self.config.copy()

            #
            # read data from analyzed fields and add to parameters
            #

            # id is only field, so we do not have to get it again from index or database
            if len(self.fields) == 1:
                parameters['id'] = docid

            # if more than id needed add fields from DB/index to parameters since that data is analysed by the plugins
            else:

                data = self.export_solr.get_data(
                    docid=docid, fields=self.fields)

                # add  to by analysed data of the first and only result to ETL/Enrichment parameters
                parameters.update(data)

            filename = docid
            # if exist delete protocol prefix file://
            if filename.startswith("file://"):
                filename = filename.replace("file://", '', 1)

            parameters['filename'] = filename

            parameters['verbose'] = self.verbose

            if self.verbose:
                print("Parameters:")
                print(parameters)

            # set markers that enriched by this plugins
            data = {}
            for plugin in self.config['plugins']:
                data['etl_' + plugin + '_b'] = True

            # start ETL / Enrichment process
            parameters, data = self.process(parameters=parameters, data=data)

        finally:

            # remove blacklisting/locking for this document, since enrichment process is now done
            self.work_in_progress_lock.acquire()

            if docid in self.work_in_progress:
                self.delete_from_work_in_progress_lock.acquire()

                self.delete_from_work_in_progress_after_commit.append(docid)

                self.delete_from_work_in_progress_lock.release()

            self.work_in_progress_lock.release()

            # set event, so main thread wakes up and knows next job/thread can be started
            self.e_job_done.set()

    #
    # get query from plugin and start enrichment process for this query
    #

    # not usable for plugin chains (i.e. extract containers like ZIP files and than OCR contents)! Use enrich_query with a compounded query instead.
    def enrich(self):

        for plugin in self.config['plugins']:

            query = "*:* AND NOT (etl_{}_b:true)".format(plugin)

            # check if plugin has own more special query and if so, use this
            module = importlib.import_module(plugin)
            objectreference = getattr(module, plugin, False)
            if objectreference:
                query = getattr(objectreference, 'query', query)

            if self.verbose:
                print("Data enrichment query: {}".format(query))

            # enrich
            self.enrich_query(query)

    #
    # Enrich the given documents by the configured plugins
    #
    # Manages the worker threads and the blacklist of ids work is in progress for,
    # so the same document is not enriched twice at the same time.
    #
    # Returns the count of documents enrichment was started for.
    #

    def enrich_documents(self, docids):

        counter = 0

        for docid in docids:

            if self.threads_max == 1:

                # no threading, do it directly in this process
                self.enrich_document(docid=docid)
                counter += 1

            else:

                #
                # Manage threading
                #

                # If doc id blacklistet (work in progress in running threads) don't start thread for docid,
                # since new search result can include some same documents again which not yet ready enriched because work goes on in a thread from step before.
                # So continue with next result.
                if docid in self.work_in_progress:
                    continue

                # wait for a job done if yet maximum threads (+1 because do not count this main thread) running
                while threading.active_count() >= self.threads_max + 1:
                    # wait for event that signals that a thread/job finished (set in enrich_document() at end)
                    # use a timeout if racing condition (setting in finished job before clearing this event here)
                    self.e_job_done.wait(1)

                # blacklist id of document work in progres
                self.work_in_progress_lock.acquire()
                self.work_in_progress.append(docid)
                self.work_in_progress_lock.release()

                # start enrichment of this document in new thread
                thread = threading.Thread(
                    target=self.enrich_document, args=(docid, ))
                self.e_job_done.clear()
                thread.start()

                counter += 1

        return counter

    #
    # delete done IDs from blacklist
    #

    def cleanup_work_in_progress(self):

        self.delete_from_work_in_progress_lock.acquire()

        while len(self.delete_from_work_in_progress_after_commit) > 0:
            docid = self.delete_from_work_in_progress_after_commit.pop()

            self.work_in_progress_lock.acquire()
            self.work_in_progress.remove(docid)
            self.work_in_progress_lock.release()

        self.delete_from_work_in_progress_lock.release()

    #
    # Wait until all started enrichment threads are done and commit their results
    #

    def finish_enrichments(self):

        # wait until all started threads done before continue (commit and end),
        while threading.active_count() > 1:
            # wait for event that signals that a thread/job finished (set in enrich_document() at end)
            # use a timeout if racing condition (setting in finished job before clearing this event here)
            self.e_job_done.wait(1)
            self.e_job_done.clear()

        # commit last results, so very last enrichments have not wait the next autocommit time (if set up)
        self.commit()

    # get ids from query
    # get fields from plugins
    # run enrichment chain with this parameters
    def enrich_query(self, query, sample_size=None, sample_field=None, sample_seed=0):

        if sample_size is None and sample_field is not None:
            raise ValueError(
                "sample_field set but no sample_size: stratified sampling needs a sample size (budget)")

        counter = 0

        solrparameters = {
            'fl': 'id',
            'rows': self.rows_per_step,
        }

        # have to proceed all documents matching this query:

        # - all yet not enriched (query for content type AND not plugin_b: true) results of enriched content type (plugin query)
        # OR
        # - container file type like ZIP archive or PST mailbox
        # - AND not yet enriched by all set plugins
        #
        # - but both cases not if no file but content of a container file (for subfiles the enrichment plugins will be runned by the run of the container plugin)

        # query matching i.e. the contenttype of to be enriched files (for example if doing OCR we do only have to process images, not all documents)
        running_plugin_query = '(' + query + ')'

        # not, if all set plugins yet runned on this document
        all_plugin_query = []
        for plugin in self.config['plugins']:
            all_plugin_query.append("(etl_{}_b:true)".format(plugin))
        all_plugin_query = '(' + ' AND '.join(all_plugin_query) + ')'

        # matching container content types like archive files
        # since our content types like f.e. images can be in containers like for example ZIP archives, so we should enrich them too)
        container_query = '(content_type:application\/zip OR id:*.zip OR content_type:application\/vnd.ms-outlook-pst OR id:*.pst)'
        # Todo for more performance:
        # distinct container_s from results with query for only needed content types instead of working trough all containers

        query = running_plugin_query + \
            ' OR (' + container_query + ' AND NOT ' + all_plugin_query + ')'

        # not try to enrich (not existent in filesystem) subfiles inside container files like ZIP archives or extracted mail attachments
        # since for subfiles the enrichment plugins will be runned by the run of the container plugin)
        query = '(' + query + ') AND NOT (container_s:*)'

        if self.verbose:
            print("Enrichment of matches the following query:")
            print(query)

        if sample_size is not None:

            # enrich only a stratified sample of the matching documents
            counter = self.enrich_sample(
                query, sample_size=sample_size, sample_field=sample_field, sample_seed=sample_seed)

        else:

            results = self.solr.search(query, **solrparameters)

            while len(results) > 0:

                counter += self.enrich_documents(
                    [result['id'] for result in results])

                # do commit, so next query wont find documents again, which were processed but not available for searcher yet
                self.commit()

                #
                # delete done IDs from blacklist
                #

                self.cleanup_work_in_progress()

                #
                # If last step (fewer search results than a step manages), wait for all threads to be done
                # before starting new search / next step (which will only find the documents again, that are work in progress in running threads and not done/marked as ready yet) to prevent unnecessary search load
                #

                if len(results) < self.rows_per_step:
                    # wait until all started threads done before continue (commit and end),
                    self.finish_enrichments()

                # are there (more) not yet enriched documents in search index for a next step?
                results = self.solr.search(query, **solrparameters)

        # if self.verbose:
        print("Enriched {} documents".format(counter))

    #
    # Enrich only a stratified sample of the documents matching the query
    #
    # Stratified by the single valued field sample_field (documents without
    # a value form an own stratum), so rare values are sampled, too.
    #

    def enrich_sample(self, query, sample_size, sample_field, sample_seed=0):

        if not sample_field:
            raise ValueError(
                "Stratified sampling (sample_size) needs a single valued field to stratify by (sample_field)")

        # get the full sample before enriching anything,
        # so the scan sees a stable index (enrichment changes and commits documents)
        strata = self.sample_ids(
            query, sample_field=sample_field, sample_size=sample_size, sample_seed=sample_seed)

        docids = []

        for key, stratum_size, sampled_docids in strata:

            if key is None:
                key = '(missing)'

            print("Stratum {}: sampled {} of {} documents".format(
                key, len(sampled_docids), stratum_size))

            docids.extend(sampled_docids)

        # enrich the sampled documents by the standard enrichment process
        counter = 0

        for i in range(0, len(docids), self.rows_per_step):

            counter += self.enrich_documents(docids[i:i + self.rows_per_step])

            self.commit()

            self.cleanup_work_in_progress()

        self.finish_enrichments()

        return counter

    #
    # Scan all documents matching the query without changing the index
    # and collect a stratified sample of document ids
    #

    def sample_ids(self, query, sample_field, sample_size, sample_seed=0):

        sampler = StratifiedSampler(budget=sample_size, seed=sample_seed)

        solrparameters = {
            'fl': 'id,{}'.format(sample_field),
            'rows': self.rows_per_step,
            # cursor needs a sort criteria including the unique key
            'sort': 'id asc',
            'cursorMark': '*',
        }

        # scan all matching documents (pysolr follows the cursor mark through all pages).
        # the index is only read, not changed (no enrichment, no commit while scanning),
        # so the scan sees a stable index
        results = self.solr.search(query, **solrparameters)

        for result in results:

            key = result.get(sample_field, None)

            # stratification field is expected to be single valued;
            # if not, use the first value in stable (sorted) order
            if isinstance(key, list):
                if key:
                    key = sorted(key)[0]
                else:
                    key = None

            sampler.add(key, result['id'])

        return sampler.sample()


# todo: export to Solr by update

if __name__ == "__main__":

    # get uri or filename from args
    from optparse import OptionParser

    parser = OptionParser("etl-enrich [options] --plugins pluginname")

    parser.add_option("-p", "--plugins", dest="plugins",
                      default=False, help="Plugins (comma separated)")
    parser.add_option("-c", "--config", dest="config",
                      default=False, help="Config file")
    parser.add_option("-q", "--query", dest="query",
                      default=False, help="Query")
    parser.add_option("--sample-size", dest="sample_size", type="int",
                      default=None, help="Enrich only a stratified sample of this size (needs --sample-field)")
    parser.add_option("--sample-field", dest="sample_field",
                      default=None, help="Single valued field to stratify the sample by (documents without a value form an own stratum)")
    parser.add_option("--sample-seed", dest="sample_seed",
                      default=0, help="Seed for the stable ranking of documents inside a stratum (same seed and same index = same sample)")
    parser.add_option("-o", "--outputfile", dest="outputfile", default=False,
                      help="Output file (if exporter set to a file format)")
    parser.add_option("-v", "--verbose", dest="verbose",
                      action="store_true", default=None, help="Print debug messages")
    (options, args) = parser.parse_args()

    etl = ETL_Enrich()

    if options.config:
        etl.read_configfile(options.config)
        etl.fields = etl.getfieldnames_from_plugins()

    if not options.verbose or options.verbose:
        etl.verbose = options.verbose

    # set (or if config overwrite) plugin config
    if options.plugins:
        etl.config['plugins'] = options.plugins.split(',')
        etl.fields = etl.getfieldnames_from_plugins()

    if options.outputfile:
        etl.config['outputfile'] = options.outputfile

    # if query, enrich IDs matching this query
    if options.query:

        etl.enrich_query(options.query,
                         sample_size=options.sample_size,
                         sample_field=options.sample_field,
                         sample_seed=options.sample_seed)

    # if no query and no id's as argument, use default query from plugins
    elif len(args) == 0:

        etl.enrich()

    # if not query but IDs
    else:

        for uri in args:

            # todo: if not local file, download to temp file if a plugin needs parameter filename

            etl.enrich_document(docid=uri)

        etl.commit()
