from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Language can be any like python, markdown

text = """
            import random

            class BankAccount:
                def __init__(self, name, balance):
                    self.name = name
                    self.balance = balance

                def deposit(self, amount):
                    self.balance += amount
                    print(f"{amount} deposited successfully.")

                def withdraw(self, amount):
                    if amount > self.balance:
                        print("Insufficient balance.")
                    else:
                        self.balance -= amount
                        print(f"{amount} withdrawn successfully.")

                def show_balance(self):
                    print(f"Account Holder: {self.name}")
                    print(f"Current Balance: {self.balance}")


            def generate_transaction():
                transactions = [100, 200, 500, 1000]
                return random.choice(transactions)


            # main program
            user = BankAccount("Gaurav", 5000)

            amount = generate_transaction()

            user.deposit(amount)
            user.withdraw(300)
            user.show_balance()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 300,  # no. of characters
    chunk_overlap = 0
)

res = splitter.split_text(text)

print(res)
print(len(res))
