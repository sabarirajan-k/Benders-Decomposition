from gurobipy import Model, GRB, quicksum

class BendersDecomposition:
    def __init__(self, fund_returns, savings_account_return, arbitary, total_budget, epsilon=0.1):
        self.fund_returns = fund_returns
        self.savings_account_return = savings_account_return
        self.number_of_funds = len(fund_returns)
        self.b = [total_budget] + [total_budget // 2] * self.number_of_funds  # Fund limits
        self.B = [1] + [0] * self.number_of_funds  # Investment in the savings account
        self.y_star = arbitary
        self.epsilon = epsilon
        self.LB = float('-inf')
        self.UB = float('inf')
        self.iteration = 0
        self.feasibility_cuts = []
        self.optimality_cuts = []

    def solve_master_problem(self):
        """Solve the master problem with the current cuts."""
        print("\n" + "="*60)
        print(f"Solving Master Problem (Iteration {self.iteration})".center(60))
        print("="*60)

        model = Model("MasterProblem")
        y = model.addVar(vtype=GRB.INTEGER, lb=0, name="y")
        z = model.addVar(vtype=GRB.CONTINUOUS, name="z")
        model.setObjective(z, GRB.MAXIMIZE)

        for dual_vars in self.feasibility_cuts:
            model.addConstr(quicksum((self.b[i] - self.B[i] * y) * dual_vars[i] for i in range(self.number_of_funds + 1)) >= 0)

        for dual_vars in self.optimality_cuts:
            model.addConstr(z <= self.savings_account_return * y + quicksum((self.b[i] - self.B[i] * y) * dual_vars[i] for i in range(self.number_of_funds + 1)))

        model.optimize()
        
        # Save and display the LP model content
        lp_filename = f"master_problem_{self.iteration}.lp"
        model.write(lp_filename)
        with open(lp_filename, "r") as lp_file:
            lp_content = lp_file.read()
            print("\nMaster Problem LP Model:\n" + "-"*60)
            print(lp_content[:300] + "\n...")  # Display first 300 characters for brevity

        return y.X, model.ObjVal

    def solve_sub_problem(self, y_star):
        """Solve the subproblem for a given y_star."""
        print("\n" + "="*60)
        print("Solving Subproblem".center(60))
        print("="*60)

        model = Model("Subproblem")
        model.setParam("InfUnbdInfo", 1)

        investments = model.addVars(self.number_of_funds, vtype=GRB.CONTINUOUS, lb=0, name="investment")
        model.addConstr(y_star + quicksum(investments[i] for i in range(self.number_of_funds)) <= self.b[0], name="total_budget")
        model.addConstrs((investments[i] <= self.b[i + 1] for i in range(self.number_of_funds)), name="max_investment_per_fund")

        model.setObjective(quicksum(self.fund_returns[i] * investments[i] for i in range(self.number_of_funds)), GRB.MAXIMIZE)
        model.optimize()

        shadow_prices = model.getAttr(GRB.Attr.Pi)
        print(f"Subproblem Status: {'Optimal' if model.Status == GRB.OPTIMAL else 'Infeasible'}")
        print(f"Subproblem Objective Value: {model.ObjVal}")
        print(f"Shadow Prices: {shadow_prices}\n")
        return shadow_prices, model.Status, model.ObjVal

    def run(self):
        """Run the Benders Decomposition algorithm."""
        while self.UB - self.LB > self.epsilon:
            self.iteration += 1
            print("\n" + "="*60)
            print(f"Iteration {self.iteration}".center(60))
            print("="*60)

            print(f"Starting Subproblem with y_star = {self.y_star}")

            dual_vars, subproblem_status, obj_value_sp = self.solve_sub_problem(self.y_star)

            if subproblem_status == GRB.INFEASIBLE:
                self.feasibility_cuts.append(dual_vars)
                cut_type = "Feasibility Cut"
            elif subproblem_status == GRB.OPTIMAL:
                self.LB = max(self.LB, self.savings_account_return * self.y_star + obj_value_sp)
                self.optimality_cuts.append(dual_vars)
                cut_type = "Optimality Cut"
            else:
                print("Subproblem has an undefined status")
                cut_type = "Undefined"

            print("\n" + "-"*60)
            print(f"Cut Added: {cut_type}".center(60))
            print(f"Lower Bound (LB): {self.LB}".center(60))
            print(f"Upper Bound (UB): {self.UB}".center(60))
            print("-"*60)

            self.y_star, self.UB = self.solve_master_problem()
            print(f"Updated y_star: {self.y_star}, New UB: {self.UB}")

        print("\n" + "="*60)
        print("Benders Decomposition Completed".center(60))
        print("="*60)
        print(f"Final amount in y: {self.y_star}".center(60))
        print(f"Final objective value: {self.UB}".center(60))
        print("="*60)

# Define problem parameters
fund_returns = [2, 6]
savings_account_return = 4
arbitary = 500
total_budget = 200

# Create BendersDecomposition instance and run the algorithm
benders = BendersDecomposition(fund_returns, savings_account_return, arbitary, total_budget)
benders.run()
