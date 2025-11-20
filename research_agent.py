import os
import time
import requests
import json
import fitz
import arxiv
import concurrent.futures
from pathlib import Path
import logging
from dotenv import load_dotenv
from urllib.parse import quote

load_dotenv()

# Configuration & Constants
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("research_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("research_scraper")

CORE_API_KEY = os.environ.get("CORE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL", "")
MAX_PAPERS = 3
DATA_DIR = Path("research_data")

DOAJ_API_URL = "https://doaj.org/api/search/articles"
CORE_API_URL = "https://api.core.ac.uk/v3/search/works"
OPENALEX_API_URL = "https://api.openalex.org/works"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Helpers: directories
def create_data_directory():
    directories = [
        DATA_DIR,
        DATA_DIR / "metadata"
    ]
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)
    return directories[0]


# Search functions
def search_arxiv(query, max_results=MAX_PAPERS):  #using arXiv
    logger.info(f"Searching arXiv for: {query}")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    try:
        results = list(client.results(search))
        logger.info(f"Found {len(results)} papers on arXiv")
        papers = []
        for paper in results:
            entry_id = paper.entry_id.split("/")[-1] if paper.entry_id else None
            papers.append({
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "abstract": paper.summary,
                "pdf_url": paper.pdf_url,
                "landing_url": f"https://arxiv.org/abs/{entry_id}" if entry_id else "",
                "published": paper.published.strftime("%Y-%m-%d") if paper.published else None,
                "id": entry_id,
                "source": "arxiv",
                "content": paper.summary
            })
        return papers
    except Exception as e:
        logger.error(f"Error searching arXiv: {e}")
        return []

def search_openalex(query, max_results=MAX_PAPERS):    #using openalex
    logger.info(f"Searching OpenAlex for: {query}")
    params = {
        "search": query,
        "filter": "has_fulltext:true",
        "per-page": max_results,
        "sort": "relevance_score:desc"
    }
    headers = {}
    if OPENALEX_EMAIL:
        headers["User-Agent"] = f"AI Research Scraper/{OPENALEX_EMAIL}"

    try:
        response = requests.get(OPENALEX_API_URL, params=params, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            logger.info(f"Found {len(results)} papers on OpenAlex")
            papers = []
            for paper in results:
                pdf_url = None
                landing = paper.get("id")
                oa = paper.get("open_access", {})
                if isinstance(oa, dict):
                    pdf_url = oa.get("oa_url")
                authors = []
                for auth in paper.get("authorships", []):
                    if auth.get("author"):
                        authors.append(auth["author"].get("display_name", "Unknown"))
                published = str(paper.get("publication_year")) if paper.get("publication_year") else None
                paper_id = paper.get("id", "").split("/")[-1]
                papers.append({
                    "title": paper.get("title", "Unknown Title"),
                    "authors": authors or ["Unknown"],
                    "abstract": paper.get("abstract", "No abstract available"),
                    "pdf_url": pdf_url,
                    "landing_url": landing,
                    "published": published,
                    "id": paper_id,
                    "source": "openalex",
                    "content": paper.get("abstract", "No abstract available")
                })
            return papers
        else:
            logger.error(f"OpenAlex API request failed with status code: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error searching OpenAlex: {e}")
        return []

def search_doaj(query, max_results=MAX_PAPERS):   #using DOAJ
    logger.info(f"Searching DOAJ for: {query}")
    encoded_query = quote(query)
    url = f"{DOAJ_API_URL}/{encoded_query}"
    params = {"pageSize": max_results}
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            logger.info(f"Found {len(results)} papers on DOAJ")
            papers = []
            for paper in results:
                bibjson = paper.get("bibjson", {})
                pdf_url = None
                for link in bibjson.get("link", []):
                    if "fulltext" in (link.get("type") or "").lower() or "pdf" in (link.get("type") or "").lower():
                        pdf_url = link.get("url")
                        break
                abstract = bibjson.get("abstract", "No abstract available")
                papers.append({
                    "title": bibjson.get("title", "Unknown Title"),
                    "authors": [author.get("name", "Unknown") for author in bibjson.get("author", [])] or ["Unknown"],
                    "abstract": abstract,
                    "pdf_url": pdf_url,
                    "landing_url": paper.get("id"),
                    "published": bibjson.get("year"),
                    "id": paper.get("id"),
                    "source": "doaj",
                    "content": abstract
                })
            return papers
        else:
            logger.error(f"DOAJ API request failed with status code: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error searching DOAJ: {e}")
        return []

def search_core(query, max_results=MAX_PAPERS):
    logger.info(f"Searching CORE for: {query}")
    if not CORE_API_KEY:
        logger.error("CORE API key not found. Please set CORE_API_KEY environment variable.")
        return []
    headers = {"Authorization": f"Bearer {CORE_API_KEY}", "Content-Type": "application/json"}
    data = {"q": query, "limit": max_results, "offset": 0}
    try:
        response = requests.post(CORE_API_URL, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            results = response.json().get("results", [])
            logger.info(f"Found {len(results)} papers on CORE")
            papers = []
            for paper in results:
                papers.append({
                    "title": paper.get("title", "Unknown Title"),
                    "authors": [author.get("name", "Unknown") for author in paper.get("authors", [])] or ["Unknown"],
                    "abstract": paper.get("abstract", "No abstract available"),
                    "pdf_url": paper.get("downloadUrl"),
                    "landing_url": paper.get("id"),
                    "published": paper.get("publishedDate", "").split("-")[0] if paper.get("publishedDate") else None,
                    "id": paper.get("id"),
                    "source": "core",
                    "content": paper.get("abstract", "No abstract available")
                })
            return papers
        else:
            logger.error(f"CORE API request failed with status code: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Error searching CORE: {e}")
        return []

"""
research_agent.py - Part 2 of 3
PDF download/extraction and strict summarization
"""
# -----------------------------
# PDF download & extraction
# -----------------------------
def download_pdf(paper, pdf_dir):
    pdf_url = paper.get("pdf_url")
    if not pdf_url:
        logger.warning(f"No PDF URL available for paper: {paper.get('title')}")
        return None
    source = paper.get("source", "unknown")
    paper_id = paper.get("id") or str(int(time.time() * 1000))
    filename = f"{source}_{paper_id}.pdf"
    local_path = pdf_dir / filename
    if local_path.exists():
        logger.info(f"PDF already exists: {local_path}")
        return local_path
    try:
        logger.info(f"Downloading PDF from {pdf_url}")
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research_scraper/1.0)"}
        resp = requests.get(pdf_url, headers=headers, stream=True, timeout=60)
        if resp.status_code == 200 and resp.content:
            with open(local_path, "wb") as f:
                f.write(resp.content)
            logger.info(f"PDF downloaded successfully: {local_path}")
            time.sleep(1)
            return local_path
        else:
            logger.error(f"Failed to download PDF. Status code: {resp.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error downloading PDF: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    if not pdf_path or not pdf_path.exists():
        return None
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return None

def process_paper(paper, pdf_dir):
    try:
        pdf_path = download_pdf(paper, pdf_dir)
        if pdf_path:
            text = extract_text_from_pdf(pdf_path)
            paper["local_pdf_path"] = str(pdf_path)
            paper["content"] = text if text else paper.get("abstract", "")
        else:
            paper["local_pdf_path"] = None
            paper["content"] = paper.get("abstract", "No content available")
        if not paper.get("landing_url"):
            if paper.get("source") == "arxiv" and paper.get("id"):
                paper["landing_url"] = f"https://arxiv.org/abs/{paper.get('id')}"
            elif paper.get("pdf_url"):
                paper["landing_url"] = paper.get("pdf_url")
            else:
                paper["landing_url"] = ""
        return paper
    except Exception as e:
        logger.error(f"Error processing paper {paper.get('title')}: {e}")
        paper["content"] = paper.get("abstract", "Error processing paper")
        return paper

# -----------------------------
# Gemini summarization (strict prompt)
# -----------------------------
def summarize_with_gemini(paper_content, title, authors, published_year, pdf_url, landing_url):
    """
    Returns:
      - plain text summary in the exact 1..7 format if paper is relevant to AI/ML/DS
      - None if Gemini call fails
      - the string "NULL" if Gemini decides the paper is not relevant (we treat as skip)
    """
    if not GEMINI_API_KEY:
        logger.error("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
        return None

    safe_authors = ', '.join(authors) if isinstance(authors, list) else (authors or "Unknown")
    safe_title = title or "Unknown Title"
    safe_published = published_year or "Not specified"
    safe_pdf = pdf_url or "Not available"
    safe_landing = landing_url or "Not available"
    safe_content_snippet = (paper_content or "")[:45000]

    prompt = f"""
You MUST summarize the paper ONLY in the following exact plain-text format.
Do NOT use JSON. Do NOT use curly braces. Do NOT use square brackets. Do NOT use code blocks.
If the paper is NOT primarily about AI, Machine Learning, Deep Learning, or Data Science, respond with
exactly: NULL
and nothing else.

FORMAT:
1. Title: {safe_title}
2. Authors: {safe_authors}
3. Published Year: {safe_published}
4. Summary: <write a clear 5-7 sentence summary here>
5. Key methods:
   a) <method 1>
   b) <method 2>
   c) <method 3>
6. Performance metrics:
   - <metric>: <value>
   - <metric>: <value>
7. Link (PDF): {safe_pdf}
   Link (Landing Page): {safe_landing}

STRICT RULES:
- NEVER output "{" or "}".
- NEVER output "[" or "]".
- NEVER output JSON.
- MUST follow numbering 1 to 7 exactly.
- If a section has no data, keep the line but leave it blank.
- The output MUST be plain text ONLY (no markdown, no code blocks).
- If the paper is not relevant to AI/ML/DS, output exactly: NULL

Paper Content:
{safe_content_snippet}
"""
    data = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    headers = {'Content-Type': 'application/json'}
    try:
        resp = requests.post(
            GEMINI_API_URL,
            json=data,
            headers=headers,
            params={'key': GEMINI_API_KEY},
            timeout=60
        )
        if resp.status_code != 200:
            logger.error(f"Error from Gemini API: {resp.status_code} - {resp.text}")
            return None

        result = resp.json()
        try:
            candidate = result.get('candidates', [])[0]
            content = candidate.get('content', {})
            parts = content.get('parts', [])
            model_text = parts[0].get('text', '').strip() if parts else candidate.get('output', '') or ''
        except Exception:
            model_text = result.get('output', '') or ''

        if not model_text:
            logger.warning("Gemini returned empty text.")
            return None

        cleaned = model_text.replace("{", "").replace("}", "").replace("[", "").replace("]", "").strip()

        if cleaned.strip().upper() == "NULL":
            return "NULL"

        if not cleaned.startswith("1. Title:"):
            header = f"1. Title: {safe_title}\n2. Authors: {safe_authors}\n3. Published Year: {safe_published}\n"
            cleaned = header + cleaned

        if "7. Link" not in cleaned and "Link (PDF):" not in cleaned:
            cleaned += f"\n7. Link (PDF): {safe_pdf}\n   Link (Landing Page): {safe_landing}"

        return cleaned
    except Exception as e:
        logger.error(f"Error calling Gemini API: {e}")
        return None

# -----------------------------
# Process papers orchestration
# -----------------------------
def process_papers_with_gemini(papers_dict):
    logger.info(f"Processing {len(papers_dict)} papers with Gemini")
    summaries = []
    relevant_papers = {}
    paper_counter = 1

    print("\n" + "=" * 80)
    print("RESEARCH PAPER SUMMARIES")
    print("=" * 80)

    for title in list(papers_dict.keys()):
        paper = papers_dict[title]
        try:
            authors = paper.get("authors", ["Unknown"])
            published_year = paper.get("published", "Unknown")
            content = paper.get("content", "") or paper.get("abstract", "")
            pdf_url = paper.get("pdf_url") or ""
            landing_url = paper.get("landing_url") or ""

            print(f"\n\nRESEARCH PAPER {paper_counter}:")
            print(f"Title: {title}")
            print(f"Source: {paper.get('source', 'Unknown')}")
            print("-" * 80)

            summary = summarize_with_gemini(content, title, authors, published_year, pdf_url, landing_url)

            if summary:
                if summary.strip() not in ["", "NULL"]:
                    # append summary_text key for compatibility with compare function
                    summaries.append({
                        "title": title,
                        "authors": authors,
                        "published_year": published_year,
                        "source": paper.get("source", "Unknown"),
                        "pdf_url": pdf_url,
                        "landing_url": landing_url,
                        "summary_text": summary
                    })
                    relevant_papers[title] = paper
                    logger.info(f"Successfully processed paper: {title}")
                    print(summary)
                else:
                    logger.info(f"Paper not relevant to AI/ML/Data Science: {title}")
                    print("This paper is not primarily focused on AI, ML, Deep Learning, or Data Science.")
                paper_counter += 1
            else:
                logger.warning(f"Failed to get summary for paper: {title}")
                print("Failed to generate summary for this paper.")
                paper_counter += 1
        except Exception as e:
            logger.error(f"Error processing paper {title} with Gemini: {e}")
            print(f"Error processing this paper: {str(e)}")
            paper_counter += 1

    print("\n" + "=" * 80)
    print(f"COMPLETED PROCESSING ")
    print("=" * 80 + "\n")
    return summaries, relevant_papers


# -----------------------------
# Compare models across papers
# -----------------------------
def compare_models_across_papers(summaries):  #picking papers based on the comparison & prompt
    
    if not GEMINI_API_KEY:
        logger.error("Gemini API key missing.")
        return None

    if not summaries:
        return "No summaries available for comparison."

    # Build combined input for the comparator prompt
    combined_text = ""
    for i, paper in enumerate(summaries, start=1):
        combined_text += f"Paper {i} Title: {paper.get('title','Unknown')}\n"
        combined_text += f"Paper {i} Summary:\n{paper.get('summary_text','')}\n\n"

    prompt = f"""
You are an expert research evaluator.

Below are summaries of several research papers.
Your task is to identify, extract, compare, and evaluate ANY type of model used in these papers.

Models include (but are not limited to):
- Machine Learning models
- Deep Learning models
- Statistical models
- Biomedical risk models
- Clinical prediction models
- Mathematical models
- Forecasting models
- Simulation models
- Analytical models
- Scoring models
- ANY computational, predictive, or formula-based model

Your tasks:
1. Extract ALL models mentioned in each paper summary (from ANY domain).
2. Compare the extracted models across the papers.
3. Select the single BEST model overall.
4. Explain WHY it is the best, based on:
   - predictive accuracy
   - interpretability
   - robustness
   - generalization
   - clinical or practical usefulness
   - validation strength
5. Output EXACTLY in this format (plain text only):

BEST MODEL:
<model name>

WHY:
<6-10 line explanation comparing it with other models>

COMPARISON:
- Paper 1 Models: <models or None>
- Paper 2 Models: <models or None>
- Paper 3 Models: <models or None>

Do NOT use JSON.
Do NOT use curly braces.
Do NOT use square brackets.
Plain text only.

{combined_text}
"""

    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        resp = requests.post(
            GEMINI_API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            params={"key": GEMINI_API_KEY},
            timeout=60
        )

        if resp.status_code != 200:
            logger.error(f"Gemini error: {resp.status_code} - {resp.text}")
            return None

        data = resp.json()

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            text = data.get("output", "") or ""

        cleaned = (
            text.replace("{", "")
                .replace("}", "")
                .replace("[", "")
                .replace("]", "")
                .strip()
        )

        return cleaned

    except Exception as e:
        logger.error(f"Error in compare_models_across_papers: {e}")
        return None

# -----------------------------
# Main
# -----------------------------
def main():
    query = input("Enter research topic to search for: ").strip()
    if not query:
        print("No query provided. Exiting.")
        return

    data_dir = create_data_directory()
    pdf_dir = None
    metadata_dir = data_dir / "metadata"

    arxiv_papers = search_arxiv(query)
    doaj_papers = search_doaj(query)
    core_papers = search_core(query)
    openalex_papers = search_openalex(query)

    all_papers = arxiv_papers + doaj_papers + core_papers + openalex_papers
    logger.info(f"Total papers found: {len(all_papers)}")

    papers_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_paper = {executor.submit(process_paper, paper, pdf_dir): paper for paper in all_papers}
        for future in concurrent.futures.as_completed(future_to_paper):
            paper = future.result()
            if paper and paper.get("title"):
                papers_dict[paper["title"]] = paper

    summaries, relevant_papers = process_papers_with_gemini(papers_dict)

    # Save metadata and summaries
    metadata_file = metadata_dir / f"{query.replace(' ', '_')}_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump({
            "query": query,
            "total_papers": len(papers_dict),
            "relevant_papers": len(relevant_papers),
            "papers": [{k: v for k, v in paper.items() if k != 'content'} for paper in papers_dict.values()],
            "summaries": summaries
        }, f, indent=2, ensure_ascii=False)

    # Compare models across generated summaries and print best-model analysis
    model_report = compare_models_across_papers(summaries)
    print("\n" + "="*80)
    print("BEST MODEL ANALYSIS")
    print("="*80)
    if model_report:
        print(model_report)
    else:
        print("Best model analysis failed or returned no result.")

    print(f"\nResearch completed")
   

if __name__ == "__main__":
    main()
