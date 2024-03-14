# !pip install casparser
import pandas as pd
from IPython.display import display, HTML
import pprint
import casparser
import json

data = casparser.read_cas_pdf("stmt3.pdf", "Fastrack@12")
# print(type(data))
print(data)
# df = pd.DataFrame.from_dict(data)
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(data)
# print(df)
# print(data.keys())

folios = len(data["folios"])
portfolio_dict = {
    "folio_id": "",
    "fund_id": "",
    "fund_name": "",
    "investment_amount": 0,
    "investment_date": "",
}
counter = 0
total = 0

df = pd.DataFrame(columns=["PAN", "Folio_number", "amfi", "isin", "scheme", "transaction_amount", "transaction_balance",
                           "transaction_date", "transaction_nav",
                           "transaction_type", "transaction_units", "statement_date", "stmt_date_nav", "value"])
for i in range(folios):
    schemes = len(data["folios"][i]["schemes"])
    #     print(i)

    if data["folios"][i]["PAN"] == "BGXPS5380R":

        for scheme in range(schemes):
            transactions = len(data["folios"][i]["schemes"][scheme]["transactions"])
            try:

                for transaction in range(transactions):
                    df.loc[len(df.index)] = [
                        data["folios"][i]["PAN"],
                        data["folios"][i]["folio"],
                        data["folios"][i]["schemes"][scheme]["amfi"],
                        data["folios"][i]["schemes"][scheme]["isin"],
                        data["folios"][i]["schemes"][scheme]["scheme"],
                        data["folios"][i]["schemes"][scheme]["transactions"][transaction]["amount"],
                        data["folios"][i]["schemes"][scheme]["transactions"][transaction]["balance"],
                        data["folios"][i]["schemes"][scheme]["transactions"][transaction]["date"],
                        data["folios"][i]["schemes"][scheme]["transactions"][transaction]["nav"],
                        data["folios"][i]["schemes"][scheme]["transactions"][transaction]["type"],
                        data["folios"][i]["schemes"][scheme]["transactions"][transaction]["units"],

                        data["folios"][i]["schemes"][scheme]["valuation"]["date"],
                        data["folios"][i]["schemes"][scheme]["valuation"]["nav"],
                        data["folios"][i]["schemes"][scheme]["valuation"]["value"],
                    ]
            except Exception as e:
                print("Exception caught :", e)

            total += data["folios"][i]["schemes"][scheme]["valuation"]["value"]
            counter += 1

print("****** Total Value : ", total, "  ***** counter : ", counter)
# display(df)
df.to_csv("stmt.csv")
