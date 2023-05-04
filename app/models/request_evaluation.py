from pydantic import BaseModel


class RequestEvaluation(BaseModel):
    loan_amnt: float
    term: str
    int_rate: float
    grade: str
    sub_grade: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    pymnt_plan: str
    purpose: str
    addr_state: str
    dti: float
    delinq_2yrs: float
    inq_last_6mths: float
    mths_since_last_delinq: float
    open_acc: float
    pub_rec: float
    revol_util: float
    initial_list_status: str
    acc_now_delinq: float

