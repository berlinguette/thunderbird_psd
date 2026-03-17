from pydantic import BaseModel, model_validator
from typing import Any

class MyModel(BaseModel):
    a: int
    b: int
    c: int

    @model_validator(mode="before")
    def compute_c_if_missing(cls, values: dict[str, Any]) -> dict[str, Any]:
        a = int(values.get("a"))
        b = int(values.get("b"))
        c = values.get("c")
        if c is None:
            values["c"] = a + b
        return values
    

if __name__ == "__main__":
    m1 = MyModel(a=2, b=3)
    print(m1.c)

    m2 = MyModel(a=2, b=3, c=10)
    print(m2.c)

    m3 = MyModel(a=2, b=3, c=None)
    print(m3.c)

    try:
        m4 = MyModel(a=2, b="foo")
        print(m4.c)
    except Exception as e:
        print(e)