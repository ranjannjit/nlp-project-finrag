cd "C:\Users\ranja\Ranjan_Doc\Ranjan\NJIT\ms\NLP\Project\project-nlp-v2"
git init
git checkout -b feature/fiqa-beir
git status
git remote add origin https://github.com/ranjannjit/nlp-project-finrag.git
# or 
git remote set-url origin https://github.com/ranjannjit/nlp-project-finrag.git   #to set new url
git add .
git commit -m "Add FiQA-BEIR"
git push -u origin feature/fiqa-beir

# Poetry manages dependencies for the project containing your Python file, rather than configuring a single file. In PowerShell, run:

cd "C:\Users\ranja\Ranjan_Doc\Ranjan\NJIT\ms\NLP\Project\project-nlp-v2\FiQA-BEIR\hybrid-retrieval"

poetry init
poetry env use python
poetry install
poetry update
poetry env activate
source C:/Users/ranja/AppData/Local/pypoetry/Cache/virtualenvs/fiqa-beir-gvTSyPQG-py3.11/Scripts/activate

To activate in interpreter:
C:\Users\ranja\AppData\Local\pypoetry\Cache\virtualenvs\fiqa-beir-gvTSyPQG-py3.11\Scripts\python.exe
"""
<!-- $ poetry show
annotated-types    0.8.0           Reusable constraint types to use with typing.Annotated
anyio              4.16.0          High-level concurrency and networking framework on top of asyncio or Trio
bm25s              0.3.16          An ultra-fast implementation of BM25 based on sparse matrices.
certifi            2026.10.23      Python package for providing Mozilla's CA Bundle.
charset-normalizer 3.5.2           The Real First Universal Charset Detector. Open, modern and actively maintained alternative to C...
click              8.5.1           Composable command line interface toolkit
cohere             7.3.1           
colorama           0.4.6           Cross-platform colored terminal text.
datasets           5.2.0           HuggingFace community-driven open-source library of datasets
dill               0.4.1           serialize all of Python
fastavro           1.13.1          Fast read/write of AVRO files
filelock           4.2.10          A platform independent file lock.
fsspec             2026.9.0        File-system specification
h11                0.16.0          A pure-Python, bring-your-own-I/O implementation of HTTP/1.1
hf-xet             1.8.0           Fast transfer of large files with the Hugging Face Hub.
httpcore           1.0.9           A minimal low-level HTTP client.
httpcore2          2.16.0          A minimal low-level HTTP client.
httpx              0.28.1          The next generation HTTP client.
httpx2             2.16.0          The next generation HTTP client.
huggingface-hub    2.4.0           Client library to download and publish models, datasets and other repos on the huggingface.co hub
idna               3.20            Internationalized Domain Names in Applications (IDNA)
jiter              0.19.0          Fast iterable JSON parser.
multiprocess       0.70.19         better multiprocessing and multithreading in Python
numpy              2.4.6           Fundamental package for array computing in Python
openai             3.51.0          The official Python library for the openai API
packaging          26.3            Core utilities for Python packages
pandas             3.1.1           Powerful data structures for data analysis, time series, and statistics
pyarrow            26.0.1          Python library for Apache Arrow
pydantic           2.14.3          Data validation using Python type hints
pydantic-core      2.50.4          Core functionality for Pydantic validation and serialization
python-dateutil    4.0.0           Extensions to the standard Python datetime module
python-dotenv      1.2.4           Read key-value pairs from a .env file and set them as environment variables
pyyaml             6.0.3           YAML parser and emitter for Python
regex              2026.11.9       Alternative regular expression module, to replace re.
requests           2.34.2          Python HTTP for Humans.
sniffio            1.3.1           Sniff out which async library your code is running under
tiktoken           0.14.0          tiktoken is a fast BPE tokeniser for use with OpenAI's models
tokenizers         0.23.3          
tqdm               4.70.1          Fast, Extensible Progress Meter
truststore         0.10.5          Verify certificates using native system trust stores
types-requests     2.33.0.20261030 Typing stubs for requests
typing-extensions  4.16.0          Backported and Experimental Type Hints for Python 3.9+
typing-inspection  0.4.5           Runtime typing introspection tools
tzdata             2026.6          Provider of IANA time zone data
urllib3            2.8.0           HTTP library with thread-safe connection pooling, file post, and more.
xxhash             4.1.0           Python binding for xxHash
""" -->

# FiQA-BEIR project overview

This project demonstrates a retrieval pipeline over the FiQA financial question-answering benchmark. It combines keyword retrieval (BM25), semantic retrieval (OpenAI embeddings), Reciprocal Rank Fusion (RRF), and Cohere reranking, then measures retrieval quality with NDCG@10.

This project overview follows the Python files in execution order. Each item explains the source statements in sequence; blank lines and comments are grouped with the surrounding code.

## Setup

Requirements: Python 3.11 or newer, Poetry, Bash (Git Bash, WSL, Linux, or macOS), internet access to download FiQA, and API keys for OpenAI and Cohere.

From PowerShell at the `FiQA-BEIR` directory:

```powershell
cd .\hybrid-retrieval
Copy-Item .env.example .env
# Edit .env and replace both placeholder values with your own API keys.
poetry install
cd ..
```

Keep `.env` private; it is ignored by Git. Never put real API keys in `.env.example` or commit them. If a real key has been exposed, revoke it and create a replacement before running the project.

To run the entire pipeline using Bash from `FiQA-BEIR`:

```bash
bash ./command.sh
```

`command.sh` installs the locked Poetry dependencies and runs the scripts in order. It downloads the dataset, creates local indexes, makes API calls for embeddings and reranking, and evaluates 50 sampled test queries. API usage may incur charges or use account quotas. The dense embedding matrix is cached, so a later run does not re-embed the corpus unless the cache is removed.

The evaluation spaces Cohere rerank calls to remain below its 10-calls-per-minute trial limit and retries rate-limit responses a limited number of times. Evaluating 50 queries therefore takes several minutes. This does not make the entire pipeline API-free: dense retrieval still uses OpenAI embeddings, including one API request per query.

For Windows PowerShell, invoke the Bash script with `bash .\command.sh` from the `FiQA-BEIR` directory. The individual Python files can also be run from `hybrid-retrieval` with `poetry run python <script>`.

## Pipeline order

1. [`download_fiqa.py`](./hybrid-retrieval/data/download_fiqa.py) fetches corpus documents, queries, and relevance judgments.
2. [`1-explore-data.py`](./hybrid-retrieval/1-explore-data.py) prints examples and checks the query/relevance data.
3. [`2-bm25.py`](./hybrid-retrieval/2-bm25.py) builds and saves the sparse keyword index.
4. [`3-embed.py`](./hybrid-retrieval/3-embed.py) embeds the corpus and saves the dense vectors.
5. [`4-rrf.py`](./hybrid-retrieval/4-rrf.py) compares BM25, dense retrieval, and their fused ranking.
6. [`5-rerank.py`](./hybrid-retrieval/5-rerank.py) reorders hybrid candidates with Cohere.
7. [`6-evaluate.py`](./hybrid-retrieval/6-evaluate.py) measures BM25, dense, hybrid, and reranked results on a sample of test queries.

Files 4–6 reuse helper implementations in [`utils/`](./hybrid-retrieval/utils/). Run the dataset download and index-building scripts before scripts that load those indexes.

## Python files explained in source order

### `data/download_fiqa.py` — download and cache the benchmark

- The module docstring describes FiQA and identifies the three BEIR data components.
- `Path` handles filesystem paths; `load_dataset` retrieves datasets from Hugging Face.
- `DATA_DIR` points to `data/fiqa` beside the script, and `mkdir(exist_ok=True)` creates that directory if it is not already present.
- The three `load_dataset` calls fetch the FiQA corpus, queries, and test relevance judgments (`qrels`).
- The three `to_parquet` calls save those datasets locally so later scripts can load them without downloading again.
- The `__main__` block prints record counts and the paths of the generated parquet files when the file is run directly.

### `1-explore-data.py` — inspect the downloaded data

- The module docstring explains the roles of a corpus document, a query, and a relevance judgment.
- `Path` and pandas are imported; `DATA_DIR` points to the local FiQA data directory.
- `read_parquet` loads the corpus, queries, and qrels. The following `print` calls show one document, one query, and one relevance row.
- `test_query_ids` converts the qrel query IDs to strings and a set for efficient membership checks.
- `test_queries` filters the query table to IDs that have relevance judgments; the next line prints the count.
- `query = test_queries.iloc[0]` selects an example, and the next prints show its ID and question text.
- The `relevant` filter finds qrel rows for that query; the last print shows the corpus IDs labeled relevant to it.

### `2-bm25.py` — build and try sparse keyword search

- The module docstring describes BM25 and its keyword-matching strengths and limitations.
- `Path`, `bm25s`, and pandas provide path, indexing, and parquet support; `DATA_DIR` and `INDEX_DIR` locate the source data and saved index.
- `read_parquet` loads the corpus; `doc_ids` and `doc_texts` extract IDs and texts in the same row order, which must be preserved for result lookup.
- The status print reports how many documents will be indexed.
- `bm25s.tokenize` converts the documents to token IDs and removes English stopwords. The next two prints show a tokenized example and sample vocabulary entries.
- `bm25s.BM25()` creates the retriever, and `retriever.index(tokens)` builds the in-memory BM25 index.
- `INDEX_DIR.mkdir(...)` creates the output folder. `retriever.save(...)` stores the index, while `doc_ids.txt` stores IDs in the matching index order.
- `search_bm25(query, k)` tokenizes a query, retrieves its top `k` results, and maps the returned integer positions back to document IDs with scores.
- The `__main__` block runs a sample financial query, looks up each result's text, and prints its rank, score, ID, and a short text preview.

### `3-embed.py` — build and query dense vectors

- The module docstring explains semantic retrieval and why it complements BM25.
- `os`, `Path`, NumPy, pandas, the OpenAI client, dotenv, and tqdm support environment variables, paths, vector operations, data loading, API calls, and progress display.
- `load_dotenv()` reads local `.env` configuration; `OpenAI(...)` creates the API client from `OPENAI_API_KEY`; `model` selects `text-embedding-3-small`.
- `DATA_DIR` and `INDEX_DIR` locate the FiQA files and dense index directory; `mkdir(...)` creates the latter.
- `embed_batch(texts)` calls the OpenAI embeddings endpoint and converts returned vectors into a float32 NumPy array.
- `build_index(doc_texts, batch_size)` iterates over document slices, embeds each batch with a progress bar, and vertically joins the vectors into one matrix.
- The corpus parquet is loaded; `doc_ids` records row order; `doc_texts` trims text and substitutes a placeholder for blank documents so each corpus row still has an embedding.
- `embeddings_path` names the cache. If it exists, the script loads it; otherwise, the script embeds the corpus and saves the resulting matrix.
- `doc_embeddings_normed` normalizes each document vector so a dot product with a normalized query vector equals cosine similarity.
- `search_dense(query, k)` embeds and normalizes one query, computes its similarity against every document, selects the highest scores, and returns document IDs with scores.
- The `__main__` block searches a sample question and prints the top results with text.

### `4-rrf.py` — combine sparse and dense rankings

- The module docstring explains why RRF combines ranks rather than incomparable raw scores.
- `defaultdict` accumulates scores; the retriever helpers load the sparse and dense indexes and the corpus; `K_RRF = 60` is the fusion smoothing constant.
- `reciprocal_rank_fusion(rankings, k)` visits each document at each 1-based rank, adds `1 / (k + rank)` to its score, and sorts documents by descending fused score.
- `BM25Retriever()`, `DenseRetriever()`, and `load_corpus()` initialize the two search systems and document table.
- `search_hybrid(query, k, candidate_k)` asks each retriever for candidates, discards their raw scores, fuses the ID rankings, and returns the first `k` fused results.
- `show(label, results)` prints a heading and up to five results, looking up each document's text in the corpus.
- The `__main__` block searches one sample query and displays BM25-only, dense-only, and hybrid rankings side by side.

### `5-rerank.py` — rerank hybrid candidates with Cohere

- The module docstring explains the cross-encoder reranking stage.
- `os`, `cohere`, dotenv, and helper imports provide API configuration, reranking, hybrid candidate generation, retrievers, and corpus access.
- `load_dotenv()` loads local keys; `cohere.ClientV2(...)` creates the client from `COHERE_API_KEY`; `RERANK_MODEL` selects the Cohere reranker.
- `BM25Retriever()` and `DenseRetriever()` load their saved indexes; `load_corpus().set_index("_id")` makes document lookup by ID direct.
- `search_reranked(query, k, candidate_k)` gets hybrid candidates, retrieves their texts, submits them with the query to Cohere, then maps Cohere's result positions back to IDs and relevance scores.
- `show(label, results)` prints each result with its text preview.
- The `__main__` block prints the un-reranked hybrid candidates and the results after the Cohere reranker.

### `6-evaluate.py` — score retrieval with NDCG@10

- The module docstring describes the evaluation metric and states that the current evaluation samples 50 queries.
- `math`, `defaultdict`, NumPy, pandas, tqdm, and helper imports support metric calculation, qrel storage, data handling, progress display, retrieval, fusion, and reranking.
- `RERANK_SAMPLE_SIZE = 50` sets the number of test queries; `SEED = 42` makes the random sample repeatable.
- `ndcg_at_k(predicted_ids, relevant, k)` computes discounted gain for the predicted ranking, computes ideal discounted gain by sorting relevance grades, and returns their ratio (or zero when there is no ideal gain).
- The query and qrel parquet files are loaded. The loop over qrel rows builds a nested mapping from query ID to document ID to relevance grade.
- `queries_with_qrels` keeps only queries with ground truth; `.sample(..., random_state=SEED)` selects the repeatable evaluation subset; the print reports its size.
- The BM25 and dense retrievers load their indexes, and `corpus_by_id` indexes document rows by ID.
- `hybrid_topk(query, candidate_k)` returns document IDs from the RRF ranking.
- `reranked_topk(query, candidate_k, k)` gets hybrid candidates, reranks them through `rerank_with_cohere`, and returns the reranked IDs.
- `results` stores one NDCG value per query and method. The evaluation loop gets each query's relevant documents, runs BM25, dense, hybrid, and hybrid-plus-rerank retrieval, then appends each method's NDCG@10.
- The final `__main__` block prints the mean NDCG@10 for each method, multiplied by 100. Note that loading and scoring the sample occur before this block; the block guards only the final table output.

### `utils/retrievers.py` — reusable BM25 and dense search

- The module docstring explains that this file contains the cleaned-up retrievers used by later scripts.
- Imports provide environment access, paths, BM25, NumPy, pandas, the OpenAI client, and dotenv.
- `load_dotenv()` loads configuration; `client` creates the OpenAI client; `ROOT`, `DATA_DIR`, `BM25_DIR`, and `DENSE_DIR` define paths relative to this helper; `EMBEDDING_MODEL` selects the query embedding model.
- `load_corpus()` reads and returns the corpus parquet as a DataFrame.
- `BM25Retriever.__init__` loads the saved BM25 index and its parallel document ID list.
- `BM25Retriever.search(query, k)` tokenizes the query, runs BM25 retrieval, and maps result positions to document IDs and float scores.
- `DenseRetriever.__init__` loads corpus IDs and cached vectors, then normalizes each vector for cosine-similarity search.
- `DenseRetriever._embed_query(query)` calls OpenAI for one query vector, converts it to float32, and normalizes it.
- `DenseRetriever.search(query, k)` computes dot-product scores, selects the largest `k`, and returns the corresponding IDs and scores.

### `utils/fusion.py` — reusable RRF and candidate generation

- The module docstring describes this as the shared RRF implementation.
- `defaultdict` accumulates scores; retriever class imports supply the expected search interfaces.
- `reciprocal_rank_fusion(rankings, k)` adds the reciprocal-rank contribution for each document in each ranking, then sorts by descending score.
- `hybrid_candidates(query, bm25, dense, candidate_k)` retrieves up to `candidate_k` IDs from each retriever, fuses them, and returns up to `candidate_k` results.

### `utils/reranker.py` — reusable Cohere reranking

- The module docstring identifies this as the shared Cohere reranker.
- Imports provide environment variables, Cohere, pandas types, dotenv, hybrid candidate generation, and the retriever classes.
- `load_dotenv()` loads configuration; `co` creates the Cohere client; `RERANK_MODEL` names its reranking model.
- `rerank_with_cohere(query, candidate_ids, corpus_by_id, k)` gathers candidate texts, sends the query and texts to Cohere, then maps the response's candidate indices back to IDs with relevance scores.
- `search_reranked(...)` obtains candidates using `hybrid_candidates` and delegates the final ordering to `rerank_with_cohere`.

### `utils/__init__.py` — package marker

This file is empty. The `utils` directory contains importable Python helper modules; this file has no runtime statements.

## Generated files and API usage

The dataset is stored under `hybrid-retrieval/data/fiqa/`; generated indexes and embeddings are stored under `hybrid-retrieval/indexes/`. These are local generated files and are excluded by the project's `.gitignore`. Keep API keys in the ignored `hybrid-retrieval/.env`, not in source files or committed configuration.



$ poetry run python 1-explore-data.py
I'm not saying I don't like the idea of on-the-job training too, but you can't expect the company to do that. Training workers is not their job - they're building software. Perhaps educational systems in the U.S. (or their students) should worry a little about getting marketable skills in exchange for their massive investment in education, rather than getting out with thousands in student debt and then complaining that they aren't qualified to do anything.
What is considered a business expense on a business trip?
query-id          8
corpus-id    566392
score             1
Name: 0, dtype: int64
Queries (test, with qrels): 648

Example query
  _id:  4641
  text: Where should I park my rainy-day / emergency fund?

  Relevant docs for this question: [44594, 406219, 319954, 397358, 88327]
(fiqa-beir-py3.11) 
ranja@Ranjan MINGW64 ~/Ranjan_Doc/Ranjan/NJIT/ms/NLP/Project/project-nlp-v2/FiQA-BEIR/hybrid-retrieval (feature/fiqa-beir)
$ poetry run python 2-bm25.py
Indexing 57638 documents with BM25...
[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 5, 12, 4, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 24, 33, 34, 35, 36, 37, 38, 39, 11, 40]]                                               
[('saying', 0), ('don', 1), ('like', 2), ('idea', 3), ('job', 4), ('training', 5), ('too', 6), ('you', 7), ('can', 8), ('expect', 9)]
                                                                                                                                                                                                                         
Query: Where should I park my rainy-day fund?

1. [ 13.76] 376148  Bond aren't necessarily any safer than the stock market.  Ultimately, there is n                                                                                                                     
2. [  8.88] 32833  In addition to the issues discussed in BrenBarn's answer, I think you need to co
3. [  7.91] 416189  For me there are two issues.  So, what to do? You have the basics of a very stro
4. [  7.83] 579848  You are only 33, with plenty of time to generate long-term returns. A correction
5. [  7.59] 580025  "I don't know Canada very well, but can offer some general points when consideri
(fiqa-beir-py3.11) 

$ poetry run python 3-embed.py
Embedding 57638 docs (~$0.22 at text-embedding-3-small)
Embedding: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 226/226 [10:08<00:00,  2.69s/it]

Query: Where should I park my rainy-day fund?

1. [0.575] 551719 "The standard low-risk/gain very-short-term parking spot these days tends to be a money market account.  However, you have only mentionedstock. For good balance, your portfolio should consider the bond market too. Consider adding a bond index fund to diversify the basic mix, taking up much of that 40%. This will also help stabilize your risk since bonds tend to move opposite stocks (prperhaps just because everyone else is also using them as themain alternative, though there are theoretical arguments why this should be so.) Eventually you may want to add a small amount of REIT fund to be mix, but that's back on the higher risk side. (By the way: Trying to guess when the next correction will occur is usually not a winning strategy; guesses tend to go wrong as often as they go right, even for pros. Rather than attempting to ""time the market"", pick a strategic mix of investments and rebalance periodically to maintain those ratios. There has been debate here about ""dollar-cost averaging"" -- see other answers -- but that idea may argue for investing and rebalancing in more small chunks rather than a few large ones. I generally actively rebalance once a year or so, and between those times let maintainng the balance suggest which fund(s) new money should go into -- minimal effort and it has worked quite well enough.,)"

2. [0.554] 32833 In addition to the issues discussed in BrenBarn's answer, I think you need to consider your medium term saving needs and existing savings.In particular, do you have a sufficient rainy day fund, a fund you will spend if things go wrong? For example, if you are dependent on a vehicle that is not covered by a guarantee or service plan, you should have enough money saved for a couple of major repairs. Depending on how secure your job is, whether it carries sick leave and long term disability, and how easy or difficult it would be to find another job in the event e.g. of your employer going bankrupt dueto a downturn in your industry, you should have months to years of minimal living expenses in your rainy day fund. If you don't have those things covered, you should urgently save as much as you can until they are covered. If you do, then the next savings priority is to put money by for retirement. Of course, if all goes well the rainy day fund will ultimately get folded into retirement, but it needs to exist now, in a form you can access quickly.

3. [0.546] 425479 "As you're saving up for an expenditure instead of investing for the long run, I would stay away from any sort of ""parking facility"" where you run the risk of not having the principal protected. The riskier investments that would potentially generate a bigger return also carry a bigger downside, ie you might not be able to get the money back that you put in. I'd shop around for a CD or a MMA/regular savings account with a half-decent interest rate. And yes, I'm aware that the return you might get is probably still less than inflation."

4. [0.543] 497993 Duffbeer703 covers most everything. The entire point of an emergency fund IS for it to be liquid.  Now I do understand (if you feel your situation requires over 6 months of living expenses): That is a lot of money to have sitting in a statement savings account! Under no circumstances should you take any sort of risk with an emergency fund. However, you COULD do this: Invest some of the assets in a six month, 1 year or 2 year CD if returns were enough to be worthwhile.  If you don't need the money, then fine, great. But if you do, you can break a Certificate of Deposit before maturity. There will bea penalty fee. You might lose interest too. But you'll have access to your money, no liquidity risk. So maybe you could put most, say 60% of your rainy dayfunds in truly liquid assets. The remainder could be in longer term CD's which you hopefully won't need because you'll be back to non rainy day living again.

5. [0.540] 527939 "The Roth vs not debate is irrelevant to the question. It doesn't matter where your emergency fund is kept, as long as it is liquid and safe. I said it before in an answer to another question: your emergency fund is not an investment -- it's your safety net This answer also says it well: an ""emergency fund"" is just that... for emergencies... NOT investment. While it ""hurts"" not to have your emergency money making more money... its MORE IMPORTANT to have quick access to it. So at TD Ameritrade, just park it in their FDIC deposit account. It will not earn any meaningful interest (at least until rates rise), but you'll be able to have access to it when you need it. Note that I would caution against putting it in a money market mutual fund. They're safer than many other investments, but they're not FDIC insured against loss and there is a potential for temporary loss of liquidity. In late 2008 when the credit markets collapsed, a lot of people suddenly became unemployed -- and needed access to their emergency funds. When Lehman Brothers went bust in September, the Reserve Primary Fund (with billions of dollars in their fund) ""broke the buck"" -- they lowered the price of shares below $1, meaning investors lost principal. The worst part is that investors were not as liquid as they wanted to be: the fund froze and it was hard to get money out. The lesson to take away from this is that one of the times you're likely to need access to your emergency fund is during a macroeconomic crisis. This is also the time when any investment that isn't guaranteed safe may potentially be (at least temporarily) unavailable or decline in value. Emergency funds should be 100% government insured. When you have your Roth funded to the point where there's extra money beyond the emergency fund, you can start investing in higher-yielding vehicles: stock or bond index ETFs would be a good start. But then that part of your Roth starts to look like a retirement account and not an emergency fund. If it were me, I'd open a Roth at a stable local bank and just keep it in their FDIC insured money market deposit account. Then if I wanted a slight boost, I might put the ""upper half"" of my emergency fund into short term CDs, but even CDs aren't worth much at the moment."

(fiqa-beir-py3.11) 

ranja@Ranjan MINGW64 ~/Ranjan_Doc/Ranjan/NJIT/ms/NLP/Project/project-nlp-v2/FiQA-BEIR/hybrid-retrieval (feature/fiqa-beir)
$ poetry run python 4-rrf.py
Query: Where should I park my rainy-day fund?
                                                                                                                                                           
BM25 only
  1. [13.7607] 376148  Bond aren't necessarily any safer than the stock market.  Ultimately, 
  2. [8.8848] 32833  In addition to the issues discussed in BrenBarn's answer, I think you 
  3. [7.9143] 416189  For me there are two issues.  So, what to do? You have the basics of a
  4. [7.8303] 579848  You are only 33, with plenty of time to generate long-term returns. A 
  5. [7.5942] 580025  "I don't know Canada very well, but can offer some general points when

Dense only
  1. [0.5705] 551719  "The standard low-risk/gain very-short-term parking spot these days te
  2. [0.5550] 32833  In addition to the issues discussed in BrenBarn's answer, I think you 
  3. [0.5477] 425479  "As you're saving up for an expenditure instead of investing for the l
  4. [0.5419] 497993  Duffbeer703 covers most everything. The entire point of an emergency f
  5. [0.5402] 527939  "The Roth vs not debate is irrelevant to the question. It doesn't matt
                                                                                                                                                           
Hybrid (RRF)
  1. [0.0323] 32833  In addition to the issues discussed in BrenBarn's answer, I think you 
  2. [0.0308] 497993  Duffbeer703 covers most everything. The entire point of an emergency f
  3. [0.0278] 376148  Bond aren't necessarily any safer than the stock market.  Ultimately, 
  4. [0.0272] 7625  For now, park it in a mix of cash and short term bond funds like the V
  5. [0.0266] 527939  "The Roth vs not debate is irrelevant to the question. It doesn't matt
(fiqa-beir-py3.11) 

ranja@Ranjan MINGW64 ~/Ranjan_Doc/Ranjan/NJIT/ms/NLP/Project/project-nlp-v2/FiQA-BEIR/hybrid-retrieval (feature/fiqa-beir)
$ poetry run python 5-rerank.py
Query: Where should I park my rainy-day fund?
                                                                                                                                                           
Hybrid (RRF) only
  1. [0.0323] 32833  In addition to the issues discussed in BrenBarn's answer, I think you 
  2. [0.0308] 497993  Duffbeer703 covers most everything. The entire point of an emergency f
  3. [0.0278] 376148  Bond aren't necessarily any safer than the stock market.  Ultimately, 
  4. [0.0272] 7625  For now, park it in a mix of cash and short term bond funds like the V
  5. [0.0266] 527939  "The Roth vs not debate is irrelevant to the question. It doesn't matt
                                                                                                                                                           
Hybrid + Cohere rerank-v4.0-fast
  1. [0.8237] 376148  Bond aren't necessarily any safer than the stock market.  Ultimately, 
  2. [0.8039] 580025  "I don't know Canada very well, but can offer some general points when
  3. [0.7723] 8088  Only having 1200 dollars sounds extremely dangerous and in my opinion 
  4. [0.7246] 285812  As others have noted, you can do better than a checking or savings acc
  5. [0.6906] 497993  Duffbeer703 covers most everything. The entire point of an emergency f
(fiqa-beir-py3.11) 

$ poetry run python 6-evaluate.py
Evaluating on 50 queries (sampled from 648)
Evaluating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [05:20<00:00,  6.40s/it]
                                                                                                                                                           
NDCG@10 x100 on FiQA (50 sampled test queries)
------------------------------------------
  BM25                   28.01
  Dense                  40.06
  Hybrid (RRF)           33.72
  Hybrid + Rerank        47.61
(fiqa-beir-py3.11) 

Q:
Where should I keep my emergency fund?
Keep your emergency fund in a safe, readily accessible cash account, such as a savings account. The sources emphasize quick access and avoiding investments that can lose value or be hard to withdraw from when you need the money. [179702] [527939]

An FDIC-insured deposit account is one option; avoid tying up the fund in investments or accounts with withdrawal barriers that might discourage you from using it in a real emergency. [527939] [64453]