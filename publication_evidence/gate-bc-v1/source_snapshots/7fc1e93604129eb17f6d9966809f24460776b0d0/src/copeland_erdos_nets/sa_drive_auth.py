"""Noninteractive, same-principal service-account authentication for Drive."""
import time

from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.auth import GoogleAuth


class ServiceAccountOnlyAuth(GoogleAuth):
    """PyDrive2 must refresh service credentials, never enter user OAuth.

    No operation retries are added: refresh/readback/permission failures escape
    to the existing durability hard gate. Tokens/key material are not recorded.
    """

    def __init__(self, key_path):
        super().__init__(settings={"save_credentials": False})
        self.credentials = ServiceAccountCredentials.from_json_keyfile_name(
            str(key_path), ["https://www.googleapis.com/auth/drive"])
        self.auth_method = "service"
        self.principal = self.credentials.service_account_email
        self.refresh_events = []

    def ServiceAuth(self):
        if self.credentials is None or self.credentials.service_account_email != self.principal:
            raise RuntimeError("Drive service-account principal changed")
        # Native PyDrive2 service decorator refreshes via credentials.refresh.
        # Unlike the user decorator, it supports SA credentials without a
        # user refresh_token and does not load client_secrets.json.
        event = {"timestamp_unix": time.time(), "principal_before": self.principal,
                 "auth_method": "service", "user_oauth_invoked": False}
        try:
            super().ServiceAuth()
            if self.auth_method != "service" or self.credentials.service_account_email != self.principal:
                raise RuntimeError("Drive service-account principal changed after refresh")
            event.update(principal_after=self.credentials.service_account_email, status="PASS")
        except BaseException:
            event["status"] = "FAIL"
            raise
        finally:
            self.refresh_events.append(event)

    def LocalWebserverAuth(self, *args, **kwargs):
        raise RuntimeError("User OAuth forbidden: LocalWebserverAuth")

    def CommandLineAuth(self, *args, **kwargs):
        raise RuntimeError("User OAuth forbidden: CommandLineAuth")

    def GetFlow(self, *args, **kwargs):
        raise RuntimeError("User OAuth forbidden: GetFlow")

    def Auth(self, *args, **kwargs):
        raise RuntimeError("User OAuth forbidden: Auth")
