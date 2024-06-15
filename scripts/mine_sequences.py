from pathlib import Path
import shutil
import faiss
from tqdm import tqdm
from muss.utils.submitit import get_executor
from muss.utils.helpers import get_file_hash, get_files_hash, log_action, yield_lines
from muss.resources.paths import get_dataset_dir
from muss.laser import get_laser_embeddings
from muss.mining.preprocessing import (
    get_subshard_paths,
    get_sentences_paths,
    create_base_index,
    get_index_name,
    sentence_tokenize_document,
    split_ccnet_shard,
    sentence_tokenize_subshard
)

from muss.mining.nn_search import (
    get_cache_dir,
    get_results_path,
    compute_and_save_nn_batched,
    get_paraphrase_pairs,
    get_pairs_path,
    compute_and_save_simplification_pairs,
    get_index_path,
    compute_and_save_embeddings,
    get_filter_string_representation,
    combine_simplifications_in_dataset,
    get_simplification_pairs_paths,
)
from muss.mining.filtering import SimplicityScorer
import gzip
import json

# Function to clear cache directories
def clear_cache_directories(dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)

# Set up stage-based directories
STAGE_DIR = Path('/home/artorius/projects/muss/stages')
def get_stage_dir(stage_name):
    stage_dir = STAGE_DIR / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir

# Function to save the stage progress
def save_stage(filepaths, stage_name):
    stage_dir = get_stage_dir(stage_name)
    for i, filepath in enumerate(filepaths):
        shutil.copy(filepath, stage_dir / f'{i:06d}.gz')

# Function to load the stage progress
def load_stage(stage_name):
    stage_dir = get_stage_dir(stage_name)
    return list(stage_dir.glob('*.gz'))

# Your original mine_sequences code...
ccnet_dir = Path(input('Please download the CCNet corpus from https://github.com/facebookresearch/cc_net and enter the path to the downloaded data: '))
language = input('What language do you want to process? (en/fr/es/pt/uk): ')
cluster = 'local'
dataset_dir = get_dataset_dir('uts') / language

slurm_partition = 'debug'
slurm_array_parallelism = 1024

# Split CCNet shards into subshards
split_stage_dir = get_stage_dir('split_ccnet_shards')
if not split_stage_dir.exists() or len(list(split_stage_dir.glob('*.gz'))) == 0:
    with log_action('Splitting CCNet shards into smaller subshards'):
        if language == 'uk':
            ccnet_filepaths = [ccnet_dir / 'uk_all_resharded.json.gz']
        else:
            n_shards = {
                'en': 15,
                'fr': 25,
                'pt': 6,
                'es': 13,
            }[language]
            ccnet_filepaths = [ccnet_dir / f'{language}_head_{i:04d}.json.gz' for i in range(n_shards)]

        raw_original_dir = dataset_dir / 'raw_original'
        raw_original_dir.mkdir(exist_ok=True, parents=True)

        if language == 'uk':
            output_dirs = [raw_original_dir / f'{language}_all']
        else:
            output_dirs = [raw_original_dir / f'{language}_head_{i:04d}' for i in range(n_shards)]

        n_docs_per_file = 50000
        executor = get_executor(cluster=cluster, slurm_partition='debug', timeout_min=1 * 30, slurm_array_parallelism=16)
        jobs = []

        print('CCNet filepaths:', ccnet_filepaths)
        print('Output directories:', output_dirs)

        with executor.batch():
            for ccnet_filepath, output_dir in zip(ccnet_filepaths, output_dirs):
                if output_dir.exists():
                    print(f'Skipping existing directory: {output_dir}')
                    continue
                job = executor.submit(split_ccnet_shard, ccnet_filepath, output_dir, n_docs_per_file)
                jobs.append(job)

        print('Job IDs for splitting:', [job.job_id for job in jobs])
        [job.result() for job in tqdm(jobs)]
        save_stage(list(raw_original_dir.glob('*.json.gz')), 'split_ccnet_shards')

# Tokenize sentences
tokenize_stage_dir = get_stage_dir('tokenize_sentences')
if not tokenize_stage_dir.exists() or len(list(tokenize_stage_dir.glob('*.txt.gz'))) == 0:
    with log_action('Tokenizing sentences'):
        executor = get_executor(
            cluster=cluster,
            slurm_partition=slurm_partition,
            timeout_min=2 * 60,
            slurm_array_parallelism=slurm_array_parallelism,
        )
        subshard_paths = get_subshard_paths(raw_original_dir)
        jobs = []

        #print('Subshard paths:', subshard_paths)

        with executor.batch():
            for i, subshard_path in enumerate(subshard_paths):
                sentences_path = dataset_dir / 'sentences' / f'{i:06d}.txt.gz'
                if sentences_path.exists():
                    continue
                sentences_path.parent.mkdir(exist_ok=True, parents=True)
                job = executor.submit(sentence_tokenize_subshard, subshard_path, sentences_path, language)
                jobs.append(job)

        print('Job IDs for tokenizing:', [job.job_id for job in jobs])
        [job.result() for job in tqdm(jobs)]
        save_stage(list(dataset_dir.glob('sentences/*.txt.gz')), 'tokenize_sentences')

embeddings_type_name = f'laser_{language}'
get_embeddings = lambda sentences: get_laser_embeddings(
    sentences, max_tokens=1000, language=language, n_encoding_jobs=4, batch_size=16
)

# Create base index
base_index_stage_dir = get_stage_dir('base_index')
if not base_index_stage_dir.exists() or len(list(base_index_stage_dir.glob('*.faiss_index'))) == 0:
    with log_action('Creating base index'):
        n_train_sentences = 10**6  # Reduced number of sentences for training
        train_sentences = []
        for sentences_path in get_sentences_paths(dataset_dir):
            for sentence in yield_lines(sentences_path):
                train_sentences.append(sentence)
                if len(train_sentences) == n_train_sentences:
                    break
            if len(train_sentences) == n_train_sentences:
                break

        base_index_dir = dataset_dir / f'base_indexes/'
        base_index_dir.mkdir(exist_ok=True, parents=True)
        base_index_path = create_base_index(
            train_sentences, get_index_name(), get_embeddings, faiss.METRIC_L2, base_index_dir
        )
        shutil.copy(base_index_path, base_index_stage_dir / base_index_path.name)

# Continue with the remaining steps as is

# Compute embeddings
with log_action('Computing embeddings'):
    cache_dir = get_cache_dir(dataset_dir) / embeddings_type_name
    indexes_dir = cache_dir / 'indexes' / f'base-index-{get_file_hash(base_index_path)}'
    indexes_dir.mkdir(exist_ok=True, parents=True)
    db_sentences_paths = get_sentences_paths(dataset_dir)
    query_sentences_paths = db_sentences_paths
    executor = get_executor(
        cluster=cluster,
        slurm_partition=slurm_partition,
        timeout_min=4 * 60,
        slurm_array_parallelism=slurm_array_parallelism,
    )
    with executor.batch():
        for sentences_path in set(query_sentences_paths + db_sentences_paths):
            if get_index_path(sentences_path, indexes_dir).exists():
                continue
            # Should take about 30 minutes each
            job = executor.submit(
                compute_and_save_embeddings, sentences_path, base_index_path, get_embeddings, indexes_dir=indexes_dir
            )
            jobs.append(job)
    print([job.job_id for job in jobs])
    [job.result() for job in tqdm(jobs)]

# Mine the paraphrases
with log_action('Mining paraphrases'):
    nn_search_results_dir = cache_dir / 'nn_search_results'
    nn_search_results_dir.mkdir(exist_ok=True, parents=True)
    topk = 8
    nprobe = 16
    executor = get_executor(
        cluster=cluster,
        slurm_partition=slurm_partition,
        timeout_min=4 * 60,
        slurm_array_parallelism=slurm_array_parallelism,
    )
    # Run NN search query by query file
    with executor.batch():
        for query_sentences_path in tqdm(query_sentences_paths, desc='submitting queries'):
            if get_results_path(query_sentences_path, db_sentences_paths, topk, nprobe, nn_search_results_dir).exists():
                continue
            # Should take about ~1h30 each
            job = executor.submit(
                compute_and_save_nn_batched,
                query_sentences_path,
                db_sentences_paths,
                topk,
                nprobe,
                indexes_dir,
                nn_search_results_dir,
                delete_intermediary=True,
            )
            jobs.append(job)
    print([job.job_id for job in jobs])
    [job.result() for job in tqdm(jobs)]

# Filter candidate paraphrases
with log_action('Filtering candidate paraphrases'):
    pairs_dir = cache_dir / 'pairs'
    pairs_dir.mkdir(exist_ok=True, parents=True)
    filter_kwargs = {
        'density': 0.6,
        'distance': 0.05,
        'levenshtein': 0.2,
        'simplicity': 0.0,
        'filter_ne': False,
    }  # Best for paraphrases
    jobs = []
    paraphrase_pairs = []
    i = 0
    is_simpler = lambda pair: True  # noqa: E731
    # Only used when mining simplifications
    if filter_kwargs.get('simplicity', 0) > 0:
        while len(paraphrase_pairs) < 10000:
            paraphrase_pairs.extend(
                get_paraphrase_pairs(
                    query_sentences_paths[i],
                    db_sentences_paths,
                    base_index_path,
                    get_embeddings,
                    cache_dir,
                    topk,
                    nprobe,
                    filter_kwargs,
                )
            )
            i += 1
        simplicity_scorer = SimplicityScorer(language=language)
        simplicity_scorer.fit(paraphrase_pairs)
        is_simpler = lambda pair: simplicity_scorer.score(*pair) > filter_kwargs['simplicity']  # noqa: E731
    executor = get_executor(
        cluster=cluster,
        slurm_partition=slurm_partition,
        timeout_min=4 * 60,
        slurm_array_parallelism=slurm_array_parallelism,
    )
    with executor.batch():
        for query_sentences_path in tqdm(query_sentences_paths, desc='query'):
            simplification_pairs_path = get_pairs_path(
                query_sentences_path, db_sentences_paths, topk, nprobe, filter_kwargs, pairs_dir
            )
            if simplification_pairs_path.exists():
                continue
            # Should take ~10 minutes
            job = executor.submit(
                compute_and_save_simplification_pairs,
                query_sentences_path=query_sentences_path,
                db_sentences_paths=db_sentences_paths,
                base_index_path=base_index_path,
                cache_dir=cache_dir,
                pairs_dir=pairs_dir,
                get_embeddings=get_embeddings,
                topk=topk,
                nprobe=nprobe,
                language=language,
                filter_kwargs=filter_kwargs,
                is_simpler=is_simpler,
            )
            jobs.append(job)
    print([job.job_id for job in jobs])
    [job.result() for job in tqdm(jobs)]

with log_action('Wrapping up paraphrases'):
    simplification_pairs = get_simplification_pairs_paths(
        query_sentences_paths, db_sentences_paths, topk, nprobe, filter_kwargs, pairs_dir
    )
    results_str = f'query-{get_files_hash(query_sentences_paths)}_db-{get_files_hash(db_sentences_paths)}_topk-{topk}_nprobe-{nprobe}'
    filter_str = get_filter_string_representation(filter_kwargs)
    dataset = f'uts_{language}_{results_str}_{filter_str}'
    print(combine_simplifications_in_dataset(simplification_pairs, dataset))