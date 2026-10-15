"""
Offline tests — no network calls. scan_jobs' fetch_* functions are thin
wrappers around `requests.get(...).json()`, so we test the parsing logic by
feeding a fixture payload through the same normalization code path.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scan_jobs import Job
from src.match import filter_and_rank, is_phd_only_title, score_job


def _jobs_from_greenhouse_fixture() -> list[Job]:
    payload = json.loads((Path(__file__).parent / "fixtures" / "greenhouse_response.json").read_text())
    jobs = []
    for item in payload["jobs"]:
        jobs.append(
            Job(
                id=f"greenhouse:examplecorp:{item['id']}",
                source="greenhouse",
                company="examplecorp",
                title=item.get("title", ""),
                location=(item.get("location") or {}).get("name", ""),
                url=item.get("absolute_url", ""),
                apply_url=item.get("absolute_url", ""),
                description=item.get("content", ""),
            )
        )
    return jobs


def test_greenhouse_parsing():
    jobs = _jobs_from_greenhouse_fixture()
    assert len(jobs) == 2
    assert jobs[0].title == "Software Engineer, New Grad"
    assert jobs[0].location == "San Francisco, CA"
    assert jobs[0].id == "greenhouse:examplecorp:4123456"
    print("test_greenhouse_parsing: PASS")


def test_match_filters_and_ranks():
    jobs = _jobs_from_greenhouse_fixture()
    matching_cfg = {
        "must_have_keywords": ["python"],
        "nice_to_have_keywords": ["python", "distributed", "new", "grad"],
        "exclude_keywords": ["senior", "staff"],
        "min_score": 0.1,
    }
    ranked = filter_and_rank(jobs, matching_cfg)
    # The senior/staff role should be excluded outright.
    assert len(ranked) == 1
    job, score = ranked[0]
    assert job.title == "Software Engineer, New Grad"
    assert score > 0
    print(f"test_match_filters_and_ranks: PASS (score={score})")


def test_exclude_matches_title_not_description():
    """The core fix: a banned word in the DESCRIPTION must not drop the job;
    only a banned word in the TITLE does."""
    cfg = {"must_have_keywords": ["engineer"], "nice_to_have_keywords": [],
           "exclude_keywords": ["senior", "manager"], "min_score": 0.0}

    # "senior"/"manager" only in the description -> KEPT (this used to be dropped)
    good = Job(id="g:1", source="greenhouse", company="c", title="Software Engineer, New Grad",
               location="", url="", apply_url="",
               description="You'll report to a senior engineer and the hiring manager.")
    # banned word in the TITLE -> dropped
    bad = Job(id="g:2", source="greenhouse", company="c", title="Senior Software Engineer",
              location="", url="", apply_url="", description="Great entry role.")
    ranked = filter_and_rank([good, bad], cfg)
    titles = [j.title for j, _ in ranked]
    assert titles == ["Software Engineer, New Grad"], titles
    print("test_exclude_matches_title_not_description: PASS")


def test_exclude_uses_word_boundaries():
    """Title excludes match whole words: 'sr' drops 'Sr.' but NOT 'SRE', and
    'lead' doesn't nuke 'Leadership'. 'II'-level roles are kept (2-4 YOE ok)."""
    cfg = {"must_have_keywords": ["engineer"], "nice_to_have_keywords": [],
           "exclude_keywords": ["senior", "sr", "snr", "lead"], "min_score": 0.0}
    jobs = [
        Job(id="1", source="g", company="c", title="Sr. Software Engineer", location="", url="", apply_url="", description="python"),
        Job(id="2", source="g", company="c", title="Site Reliability Engineer (SRE)", location="", url="", apply_url="", description="python"),
        Job(id="3", source="g", company="c", title="Software Engineer II", location="", url="", apply_url="", description="python"),
        Job(id="4", source="g", company="c", title="Lead Engineer", location="", url="", apply_url="", description="python"),
        Job(id="5", source="g", company="c", title="Software Engineer, New Grad", location="", url="", apply_url="", description="python"),
    ]
    kept = {j.id for j, _ in filter_and_rank(jobs, cfg)}
    assert kept == {"2", "3", "5"}, kept   # SRE, II, and New Grad survive; Sr. and Lead dropped
    print("test_exclude_uses_word_boundaries: PASS")


def test_work_authorization_excludes():
    """Only HARD bars (citizen / green-card / clearance / US-person) drop. Plain
    'no visa sponsorship' roles are KEPT -- an F-1/OPT candidate can work those --
    and positive offers + EEO boilerplate must never be dropped."""
    bars = ["u.s. citizen", "us citizen", "green card holder", "u.s. person",
            "export control", "security clearance", "ts/sci", "polygraph"]
    cfg = {"must_have_keywords": ["engineer"], "nice_to_have_keywords": [],
           "exclude_keywords": [], "exclude_description_keywords": bars, "min_score": 0.0}
    def dropped(desc):
        j = Job(id="x", source="g", company="c", title="Software Engineer", location="",
                url="", apply_url="", description="python engineer. " + desc)
        return score_job(j, cfg["must_have_keywords"], cfg["nice_to_have_keywords"],
                         cfg["exclude_keywords"], cfg["exclude_description_keywords"]) is None
    # hard bars -> dropped
    assert dropped("Must be a U.S. citizen or green card holder.")
    assert dropped("An active security clearance is required.")
    assert dropped("This role is export control restricted to US persons.")
    assert dropped("Requires an active TS/SCI clearance with polygraph.")  # defense/intel
    # F-1/OPT can still do these -> KEPT
    assert not dropped("We cannot sponsor work visas for this role.")     # no-sponsorship: keep
    assert not dropped("Must be authorized to work without sponsorship.")  # no-sponsorship: keep
    assert not dropped("We offer visa sponsorship and relocation.")        # positive offer: keep
    assert not dropped("Considered without regard to citizenship status, race, or religion.")  # EEO: keep
    print("test_work_authorization_excludes: PASS")


def test_phd_only_title_filter():
    """PhD-only titles drop when enabled; MS/PhD and non-PhD titles survive
    (an MS candidate is eligible for those)."""
    assert is_phd_only_title("2027 Summer Intern, PhD, Software Engineer")
    assert is_phd_only_title("Research Scientist, Ph.D.")
    assert not is_phd_only_title("2027 Summer Intern, MS/PhD, Software Engineer")
    assert not is_phd_only_title("PhD or MS Machine Learning Intern")
    assert not is_phd_only_title("Software Engineer, New Grad")

    cfg = {"must_have_keywords": ["engineer"], "nice_to_have_keywords": [],
           "exclude_keywords": [], "exclude_phd_only_titles": True, "min_score": 0.0}
    def job(title):
        return Job(id=title, source="g", company="c", title=title, location="",
                   url="", apply_url="", description="python engineer")
    kept = {j.title for j, _ in filter_and_rank(
        [job("PhD Research Engineer"), job("MS/PhD Intern Engineer"),
         job("New Grad Software Engineer")], cfg)}
    assert kept == {"MS/PhD Intern Engineer", "New Grad Software Engineer"}, kept
    print("test_phd_only_title_filter: PASS")


def test_exclude_description_keywords():
    """Experience bars live in the body -> exclude_description_keywords drops them."""
    cfg = {"must_have_keywords": ["engineer"], "nice_to_have_keywords": [],
           "exclude_keywords": [], "exclude_description_keywords": ["5+ years"], "min_score": 0.0}
    junior = Job(id="j:1", source="greenhouse", company="c", title="Backend Engineer",
                 location="", url="", apply_url="", description="New grads welcome.")
    experienced = Job(id="j:2", source="greenhouse", company="c", title="Backend Engineer",
                      location="", url="", apply_url="", description="Requires 5+ years of experience.")
    ranked = filter_and_rank([junior, experienced], cfg)
    assert [j.id for j, _ in ranked] == ["j:1"]
    print("test_exclude_description_keywords: PASS")


def test_rerank_by_ats():
    """Jobs are re-ordered by how well the master resume matches; thin/weak
    postings are dropped by the thresholds."""
    from src.match import rerank_by_ats
    master = {"skills": [{"category": "x", "entries": "Python, SQL, FastAPI, PyTorch"}]}
    strong = Job(id="a", source="greenhouse", company="c", title="Backend Engineer",
                 location="", url="", apply_url="",
                 description="Python, SQL, FastAPI, PyTorch, Docker needed.")  # 5 named, 4 have -> 80%
    partial = Job(id="b", source="greenhouse", company="c", title="Platform Engineer",
                  location="", url="", apply_url="",
                  description="Python, SQL, Docker, Kubernetes, AWS, Terraform.")  # 6 named, 2 have -> 33%
    thin = Job(id="c", source="greenhouse", company="c", title="Backend Engineer",
               location="", url="", apply_url="",
               description="We want Python and good communication.")  # 1 skill -> below min_jd_skills
    ranked = [(partial, 0.9), (strong, 0.5), (thin, 0.99)]
    out = rerank_by_ats(ranked, master, min_ats=50, min_jd_skills=4)
    assert [j.id for j, _ in out] == ["a"], out   # thin (too few) + partial (<50%) dropped
    # thresholds off except thin filter: strong (80%) still ranks above partial (33%)
    out2 = rerank_by_ats(ranked, master, min_ats=0, min_jd_skills=4)
    assert out2[0][0].id == "a"
    print("test_rerank_by_ats: PASS")


def test_role_tier_and_prefer_engineering():
    from src.match import role_tier, rerank_by_ats
    assert role_tier("Software Engineer, Backend") == 2
    assert role_tier("Machine Learning Fellow") == 2       # "machine learning"
    assert role_tier("Data Engineer") == 2
    assert role_tier("DevOps Engineer") == 2
    assert role_tier("Site Reliability Engineer (SRE)") == 2
    assert role_tier("Cloud Engineer") == 2
    assert role_tier("Security Engineer") == 2
    assert role_tier("Design Engineer") == 1               # generic engineer
    assert role_tier("AI Strategy Consultant") == -1
    assert role_tier("Infrastructure Capacity Planner") == -1
    assert role_tier("Field Engineer, Public Sector") == -1  # customer-facing despite 'engineer'
    assert role_tier("Program Coordinator") == 0

    master = {"skills": [{"category": "x", "entries": "Python, SQL"}]}
    swe = Job(id="a", source="g", company="c", title="Backend Engineer", location="", url="", apply_url="",
              description="Python, SQL, Docker, Kubernetes, AWS, Java")          # tier 2
    biz = Job(id="b", source="g", company="c", title="AI Strategy Consultant", location="", url="", apply_url="",
              description="Python, SQL, data analysis, machine learning, statistics, communication")  # tier -1
    out = rerank_by_ats([(biz, 0.9), (swe, 0.5)], master, prefer_engineering=True)
    assert [j.id for j, _ in out] == ["a"], out   # consultant dropped, SWE kept
    print("test_role_tier_and_prefer_engineering: PASS")


if __name__ == "__main__":
    test_greenhouse_parsing()
    test_match_filters_and_ranks()
    test_exclude_matches_title_not_description()
    test_exclude_uses_word_boundaries()
    test_work_authorization_excludes()
    test_phd_only_title_filter()
    test_exclude_description_keywords()
    test_rerank_by_ats()
    test_role_tier_and_prefer_engineering()
    print("\nAll tests passed.")
