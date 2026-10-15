import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from fastapi import HTTPException

from services import supabase_auth
from services import workspace as workspace_service
from services.supabase_auth import CurrentUser


class WorkspaceIsolationTests(unittest.TestCase):
    def test_user_state_and_artifacts_use_separate_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_settings = SimpleNamespace(
                storage_dir=root,
                state_path=root / "state.json",
                upload_dir=root / "uploads",
                artifact_dir=root / "artifacts",
            )
            with patch.object(workspace_service, "settings", fake_settings):
                workspace_service._workspaces.clear()
                first_id, second_id = str(uuid4()), str(uuid4())
                first = workspace_service.workspace_for(CurrentUser(first_id, "one@example.test", "token-one", True))
                second = workspace_service.workspace_for(CurrentUser(second_id, "two@example.test", "token-two", True))

                first.state_store.add_dataset({"id": "ds_private", "filename": "private.csv"})

                self.assertIsNotNone(first.state_store.get_dataset("ds_private"))
                self.assertIsNone(second.state_store.get_dataset("ds_private"))
                self.assertNotEqual(first.upload_dir, second.upload_dir)
                self.assertNotEqual(first.artifact_dir, second.artifact_dir)
                self.assertNotIn("access_token", first.__dict__)
                workspace_service._workspaces.clear()

    def test_required_auth_fails_closed_without_supabase_configuration(self):
        fake_settings = SimpleNamespace(supabase_url="", supabase_publishable_key="", auth_required=True)
        with patch.object(supabase_auth, "settings", fake_settings):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(supabase_auth.get_current_user(authorization=None))
        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
