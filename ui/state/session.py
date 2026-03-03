from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionState:
    user_id: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str = "learner"
    token: Optional[str] = None

    def is_authenticated(self) -> bool:
        return bool(self.user_id and self.token)
