class User:
    def __init__(
        self,
        id: str,
        name: str,
        email: str,
        role: str,
    ):
        self.id = id
        self.name = name
        self.email = email
        self.role = role

from pydantic import BaseModel

class User(BaseModel):
    id: str
    name: str
    email: str
    role: str