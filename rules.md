# Python Developer Guideline

## 1. Typing

- Type all public functions, methods, and class attributes.
- Use `| None` for optional types (we target Python 3.10+).
- Do **not** annotate trivial local variables (e.g., `i = 0`, `name = "abc"`).

### ✅ Do

```python
def find_user(user_id: str) -> User | None:
    ...

self.timeout: float
self.users: dict[str, User]
```

### ❌ Don’t

```python
count: int = 0         # unnecessary
name: str = "default"  # unnecessary
```

---

## 2. Docstrings

- Use **multiline Google-style** docstrings for all public classes and methods.
- Do **not** use one-line docstrings, even for simple functions.
- Always describe **what the method does**, **each argument**, and **what it returns**.
- Use complete sentences with punctuation and consistent indentation.

### ✅ Example

```python
def login(username: str, password: str) -> bool:
    """
    Attempt to authenticate a user with the given credentials.

    Args:
        username (str): The user's login name.
        password (str): The user's password.

    Returns:
        bool: True if authentication is successful, False otherwise.
    """
```

---

## 3. Class Design

- Always call `super().__init__()` if the base class defines `__init__`.
- Type all attributes explicitly.
- Keep constructors free of complex logic; initialization should be simple and predictable.

---

## 4. Minimal Full Example

```python
class EmailClient:
    """
    A simple email-sending client.

    Attributes:
        smtp_host (str): SMTP server hostname.
        port (int): Port to connect to.
        sender (str | None): Optional default sender address.
    """

    smtp_host: str
    port: int
    sender: str | None

    def __init__(self, smtp_host: str, port: int, sender: str | None = None) -> None:
        """
        Initialize the email client.

        Args:
            smtp_host (str): Hostname of the SMTP server.
            port (int): Port number.
            sender (str | None): Default sender email address.
        """
        self.smtp_host = smtp_host
        self.port = port
        self.sender = sender

    def send(self, to: str, subject: str, body: str) -> bool:
        """
        Send an email message.

        Args:
            to (str): Recipient address.
            subject (str): Email subject.
            body (str): Email content.

        Returns:
            bool: True if the message is sent successfully, False otherwise.
        """
        pass
```
