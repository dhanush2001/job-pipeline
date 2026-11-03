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


def test_graduation_window_parsing():
    from src.match import graduation_window
    # The Databricks miss: a later-graduating internship.
    assert graduation_window("Must be graduating in fall 2027 or spring 2028.") == (2027, 2028)
    assert graduation_window("Expected graduation between December 2026 and June 2027.") == (2026, 2027)
    assert graduation_window("Candidates graduating in 2027 or later.") == (2027, 9999)
    assert graduation_window("You must graduate by June 2027.") == (0, 2027)
    assert graduation_window("Class of 2026 preferred.") == (2026, 2026)
    # No graduation-DATE language -> no requirement (None), so nothing is dropped.
    assert graduation_window("Join our 2027 New Graduate program. Python, SQL.") is None
    assert graduation_window("Founded in 2015, we build data tools.") is None
    print("test_graduation_window_parsing: PASS")


def test_graduation_screen_drops_ineligible_keeps_eligible():
    from src.match import graduation_mismatch
    cfg = {"must_have_keywords": ["intern"], "nice_to_have_keywords": [],
           "exclude_keywords": [], "min_score": 0.0, "graduation_year": 2026}

    def job(i, desc):
        return Job(id=i, source="g", company="c", title="Software Engineer Intern",
                   location="", url="", apply_url="", description=desc)
    jobs = [
        job("databricks", "Python, Spark. Must be graduating in fall 2027 or spring 2028."),
        job("window-ok", "Python. Expected graduation between December 2026 and 2027."),
        job("newgrad-start", "2027 New Graduate start date. Python, SQL."),   # start date, not grad req
        job("no-req", "Python, SQL, distributed systems."),
    ]
    kept = {j.id for j, _ in filter_and_rank(jobs, cfg)}
    assert kept == {"window-ok", "newgrad-start", "no-req"}, kept
    # Unknown grad year -> never drop.
    assert graduation_mismatch("graduating fall 2027 or spring 2028", None) is None
    # Toggle off -> keep everything (the internship screen would also drop the
    # "graduating fall 2027" intern, so it goes off too).
    off = filter_and_rank(jobs, {**cfg, "exclude_graduation_mismatch": False,
                                 "exclude_post_graduation_internships": False})
    assert len(off) == 4
    print("test_graduation_screen_drops_ineligible_keeps_eligible: PASS")


def test_required_experience_parsing():
    from src.match import required_experience_years as r
    # The live Stripe miss: master's plus a spelled-out three years.
    assert r("Master's degree plus three (3) years of professional software engineering experience.") == 3
    assert r("3+ years of experience building backend services.") == 3
    assert r("Requires 3-5 years of industry experience.") == 3
    assert r("At least two years of experience with SQL.") == 2
    assert r("Experience: 4+ years in distributed systems.") == 4
    # Alternatives: the lowest path applies.
    assert r("BS with 3+ years of experience, or MS with 1+ year of experience.") == 1
    assert r("0-2 years of experience. New grads welcome.") == 0
    # Not requirements.
    assert r("2+ years of Python experience preferred.") is None
    assert r("A 4-year degree in Computer Science.") is None
    assert r("Founded 10 years ago, we serve 2023 customers.") is None
    # Phrasings without the word "experience" right after the number.
    assert r("2+ years of relevant industry experience (e.g. data/ML scientist, "
             "tech lead, junior faculty) and a Master's degree or PhD.") == 2
    assert r("Minimum 2 years in software development.") == 2
    assert r("2+ yrs with Python.") == 2
    assert r("At least 2 years working as a data scientist.") == 2
    assert r("You have 2+ years building production systems.") == 2
    assert r("2+ years in a tech lead role.") == 2
    # Time spans, not requirements.
    assert r("We have grown 3x in the last 2 years.") is None
    assert r("You will be promoted within 2 years.") is None
    assert r("Over the past 5 years we built our platform.") is None
    assert r("This is a 2 year contract.") is None
    print("test_required_experience_parsing: PASS")


def test_internship_after_graduation_screen():
    from src.match import internship_after_graduation as late, filter_and_rank
    assert late("2027 Summer Intern, BS/MS, Embedded, Software Engineer", "", 2026)
    assert late("Fall 2027 Co-op, ML", "", 2026)
    assert late("Software Engineer Intern", "You will return to school after the internship.", 2026)
    assert late("Software Engineering Intern - Summer 2026", "", 2026) is None
    assert late("Software Engineer Intern", "A paid 12-week internship.", 2026) is None
    # Enrollment required after you graduate (Dec 2026) -> drop.
    assert late("Software Engineer Intern", "This internship runs Summer 2027 in Seattle.", 2026)
    assert late("ML Intern", "Must be enrolled in a degree program through Spring 2027.", 2026)
    assert late("ML Intern", "You must be currently enrolled and graduating in May 2027.", 2026)
    assert late("ML Intern", "Must remain enrolled in your program during the internship (Summer 2027).", 2026)
    # Enrollment / dates that end by December 2026 are fine.
    assert late("ML Intern", "Must be currently enrolled in a Master's program; Fall 2026 term.", 2026) is None
    assert late("ML Intern", "Open to students graduating December 2026.", 2026) is None
    assert late("ML Intern", "We were founded in 2027 and serve 2028 customers.", 2026) is None, "years with no term/enrollment context are ignored"
    assert late("2027 New Grad Software Engineer", "", 2026) is None, "new-grad start dates are fine"
    assert late("International Software Engineer", "", 2026) is None, "'intern' must be a whole word"
    jobs = [Job(id=i, source="g", company="c", title=t, location="", url="", apply_url="",
                description="python software engineer")
            for i, t in [("1", "2027 Summer Intern, Software Engineer"), ("2", "Software Engineer, New Grad")]]
    cfg = {"must_have_keywords": ["python"], "graduation_year": 2026}
    kept = [j.title for j, _ in filter_and_rank(jobs, cfg)]
    assert kept == ["Software Engineer, New Grad"], kept
    cfg["exclude_post_graduation_internships"] = False
    assert len(filter_and_rank(jobs, cfg)) == 2, "switch turns the screen off"
    print("test_internship_after_graduation_screen: PASS")


def test_experience_screen():
    cfg = {"must_have_keywords": ["engineer"], "nice_to_have_keywords": [],
           "exclude_keywords": [], "min_score": 0.0, "max_required_experience_years": 2}

    def job(i, desc):
        return Job(id=i, source="g", company="c", title="Software Engineer",
                   location="", url="", apply_url="", description=desc)
    jobs = [job("stripe", "Master's plus three (3) years of experience."),
            job("newgrad", "0-2 years of experience. New grads welcome."),
            job("soft", "3+ years of experience preferred but not required."),
            job("none", "Python, SQL, distributed systems.")]
    kept = {j.id for j, _ in filter_and_rank(jobs, cfg)}
    assert kept == {"newgrad", "soft", "none"}, kept
    # Unset -> no experience screening at all.
    assert len(filter_and_rank(jobs, {k: v for k, v in cfg.items() if k != "max_required_experience_years"})) == 4
    print("test_experience_screen: PASS")


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
    test_internship_after_graduation_screen()
    test_greenhouse_parsing()
    test_match_filters_and_ranks()
    test_exclude_matches_title_not_description()
    test_exclude_uses_word_boundaries()
    test_work_authorization_excludes()
    test_phd_only_title_filter()
    test_graduation_window_parsing()
    test_graduation_screen_drops_ineligible_keeps_eligible()
    test_required_experience_parsing()
    test_experience_screen()
    test_exclude_description_keywords()
    test_rerank_by_ats()
    test_role_tier_and_prefer_engineering()
    print("\nAll tests passed.")


def test_adzuna_errors_never_log_keys():
    from src.scan_jobs import _redact
    err = Exception("Max retries exceeded with url: /v1/api/jobs/us/search/1?app_id=abc123&app_key=s3cr3t&what=x")
    out = _redact(err)
    assert "abc123" not in out and "s3cr3t" not in out
    assert "what=x" in out


def test_undergrad_only_titles_are_dropped_for_ms_candidate():
    from src.match import is_undergrad_only_title
    assert is_undergrad_only_title("Campus Undergraduate Full-Time Engineer - 2027 AI Engineer I, Enterprise Technology Services- New York, NY")
    assert is_undergrad_only_title("Bachelor's New Grad Software Engineer") is False  # "grad" keeps it
    assert not is_undergrad_only_title("Campus Graduate Masters Full-Time Engineer - 2027 AI Engineer I")
    assert not is_undergrad_only_title("Software Engineer, BS/MS 2027")
    assert not is_undergrad_only_title("Undergraduate or Graduate Software Engineer Intern")
    assert not is_undergrad_only_title("University Grad Software Engineer 2027 (USA)")


def test_phd_required_screen_keeps_ms_eligible_research_roles():
    from src.match import phd_required
    assert phd_required("<li>PhD in Computer Science or a related field is required.</li>")
    assert phd_required("Candidates must have a Ph.D. in machine learning.")
    assert not phd_required("PhD preferred. Strong Python skills required.")
    assert not phd_required("MS or PhD in Computer Science required.")
    assert not phd_required("PhD or equivalent research experience required.")
    assert not phd_required("Open to PhD candidates enrolled in a doctoral program; must be returning to school.")
    assert not phd_required("Our team includes many PhDs.")


def test_ai_research_titles_rank_as_core_roles():
    from src.match import role_tier
    for t in ("Researcher, Models", "Research Engineer, Post-Training Inference",
              "Member of Technical Staff (ML)", "Research Assistant, Computational Biology"):
        assert role_tier(t) == 2, t
    assert role_tier("UX Researcher") == -1
    assert role_tier("Market Research Analyst") == -1
    assert role_tier("Developer Relations") == -1


def test_it_support_titles_are_kept_not_demoted():
    from src.match import role_tier
    for t in ("IT Support Specialist", "Help Desk Technician", "IT Support Engineer", "Desktop Support Analyst"):
        assert role_tier(t) == 1, t
    assert role_tier("Customer Support Specialist") == -1
    assert role_tier("Software Engineer") == 2


def test_it_support_roles_use_their_own_ats_floor(monkeypatch):
    import src.match as mt
    monkeypatch.setattr(mt, "score_resume", lambda m, d: {"score": 40})
    monkeypatch.setattr(mt, "requirements", lambda d: list(range(8)))
    mk = lambda t: mt.Job(id=t, source="greenhouse", company="c", title=t, location="", url="", apply_url="", description="")
    jobs = [(mk("IT Support Engineer"), 0.1), (mk("Software Engineer"), 0.1)]
    kept = mt.rerank_by_ats(jobs, {}, min_ats=60, min_jd_skills=6, prefer_engineering=True, min_ats_it=35)
    assert [j.title for j, _ in kept] == ["IT Support Engineer"]
