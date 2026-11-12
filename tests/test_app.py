import time
import unittest
from pathlib import Path

from app import create_app
from tests.helpers import FakeMotionModel, default_spec, make_settings, tmpdir, write_video
from violence_model import ViolenceModel


def wait_done(client, job_id, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(f"/api/jobs/{job_id}").get_json()
        if body["state"] in ("done", "error"):
            return body
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def wait_empty(directory: Path, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        if not [p for p in directory.iterdir() if p.name != ".gitkeep"]:
            return True
        time.sleep(0.05)
    return False


class AppTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tmpdir()
        self.tmp = Path(self._tmp.name)
        self.settings = make_settings(self.tmp, max_upload_mb=1)
        self.model = FakeMotionModel()
        self.app = create_app(self.settings, model=self.model, spec=default_spec())
        self.app.testing = False  # exercise the real error handlers
        self.client = self.app.test_client()
        self.motion, self.still = self.tmp / "motion.avi", self.tmp / "still.avi"
        write_video(self.motion, seconds=3, size=(64, 48), kind="motion")
        write_video(self.still, seconds=3, size=(64, 48), kind="static")

    def tearDown(self):
        self._tmp.cleanup()

    def post(self, path, url="/api/analyze", name=None):
        data = {"video": (open(path, "rb"), name or Path(path).name)}
        return self.client.post(url, data=data, content_type="multipart/form-data")

    # --- pages
    def test_home_and_static_assets(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"SentinelAI", r.data)
        for url in ("/static/css/style.css", "/static/js/app.js"):
            self.assertEqual(self.client.get(url).status_code, 200, url)

    def test_health(self):
        body = self.client.get("/api/health").get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["model"]["state"], "ready")
        self.assertEqual(body["settings"]["allowed_extensions"], ["mp4", "avi", "mov"])

    def test_debug_is_off(self):
        self.assertFalse(self.app.debug)
        self.assertFalse(self.settings.debug)

    # --- async flow
    def test_upload_analyse_and_cleanup(self):
        r = self.post(self.motion)
        self.assertEqual(r.status_code, 202)
        job = wait_done(self.client, r.get_json()["job_id"])
        self.assertEqual(job["state"], "done")
        self.assertEqual(job["result"]["label"], "violence")
        self.assertEqual(job["progress"]["percent"], 100.0)
        self.assertTrue(wait_empty(self.settings.upload_dir), "uploaded video was not deleted")

        r2 = self.post(self.still)
        job2 = wait_done(self.client, r2.get_json()["job_id"])
        self.assertEqual(job2["result"]["label"], "non_violence")
        self.assertTrue(wait_empty(self.settings.upload_dir))

    def test_repeated_predictions_are_stable(self):
        outs = []
        for _ in range(3):
            job = wait_done(self.client, self.post(self.motion).get_json()["job_id"])
            outs.append(job["result"]["violence_probability"])
        self.assertEqual(len(set(outs)), 1)
        self.assertEqual(self.model.calls, 3)  # one cached model instance reused

    def test_sync_predict_endpoint(self):
        r = self.post(self.motion, url="/predict")
        body = r.get_json()
        self.assertEqual(r.status_code, 200)
        self.assertTrue(body["ok"])
        self.assertEqual(body["prediction"], "Violence")
        self.assertIn("confidence", body)
        self.assertTrue(wait_empty(self.settings.upload_dir))

    # --- validation / errors
    def test_no_file(self):
        r = self.client.post("/api/analyze", data={}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"]["code"], "no_file")

    def test_bad_extension(self):
        txt = self.tmp / "notes.txt"
        txt.write_text("hello")
        r = self.post(txt)
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"]["code"], "bad_extension")
        self.assertTrue(wait_empty(self.settings.upload_dir))

    def test_corrupted_video(self):
        bad = self.tmp / "broken.mp4"
        bad.write_bytes(bytes(range(256)) * 40)
        r = self.post(bad)
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.get_json()["error"]["code"], "invalid_video")
        self.assertTrue(wait_empty(self.settings.upload_dir))

    def test_empty_file(self):
        empty = self.tmp / "empty.mp4"
        empty.write_bytes(b"")
        self.assertEqual(self.post(empty).status_code, 422)

    def test_file_too_large(self):
        big = self.tmp / "big.mp4"
        big.write_bytes(b"\0" * (3 * 1024 * 1024))  # beyond Flask's hard limit
        self.assertEqual(self.post(big).status_code, 413)
        mid = self.tmp / "mid.mp4"
        mid.write_bytes(b"\0" * int(1.5 * 1024 * 1024))  # over the app limit, under Flask's headroom
        r = self.post(mid)
        self.assertEqual(r.status_code, 413)
        self.assertEqual(r.get_json()["error"]["code"], "too_large")
        self.assertTrue(wait_empty(self.settings.upload_dir))

    def test_unknown_job_and_unknown_route(self):
        self.assertEqual(self.client.get("/api/jobs/deadbeef").status_code, 404)
        r = self.client.get("/nope")
        self.assertEqual(r.status_code, 404)
        self.assertFalse(r.get_json()["ok"])

    def test_stale_uploads_are_purged_on_start(self):
        stale = self.settings.upload_dir / "old.mp4"
        stale.write_bytes(b"x")
        import os
        os.utime(stale, (time.time() - 99999, time.time() - 99999))
        create_app(self.settings, model=self.model, spec=default_spec())
        self.assertFalse(stale.exists())


class MissingModelTests(unittest.TestCase):
    def test_real_model_without_weights_returns_clear_503(self):
        with tmpdir() as d:
            tmp = Path(d)
            settings = make_settings(tmp)
            spec = default_spec()
            model = ViolenceModel(spec, settings)  # real class, no weights downloaded
            app = create_app(settings, model=model, spec=spec)
            client = app.test_client()
            health = client.get("/api/health").get_json()
            self.assertIn(health["model"]["state"], ("weights_missing", "dependency_missing"))
            self.assertFalse(health["model"]["available"])
            vid = tmp / "v.avi"
            write_video(vid, seconds=2, size=(64, 48))
            r = client.post("/api/analyze", data={"video": (open(vid, "rb"), "v.avi")}, content_type="multipart/form-data")
            self.assertEqual(r.status_code, 503)
            err = r.get_json()["error"]
            self.assertIn(err["code"], ("weights_missing", "dependency_missing"))
            self.assertTrue(err["message"])
            self.assertEqual([p for p in settings.upload_dir.iterdir() if p.name != ".gitkeep"], [])

    def test_stand_in_with_missing_weights_state(self):
        with tmpdir() as d:
            settings = make_settings(Path(d))
            app = create_app(settings, model=FakeMotionModel(state="weights_missing"), spec=default_spec())
            r = app.test_client().post("/predict", data={}, content_type="multipart/form-data")
            self.assertEqual(r.status_code, 503)


if __name__ == "__main__":
    unittest.main()
