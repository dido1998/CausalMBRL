def get_stats(rule_selections, variable_selections, variable_rules, application_option, num_rules = 5, num_blocks = 1):
    if isinstance(application_option, str):
        application_option = int(application_option.split('.')[0])
    for b in range(rule_selections[0].shape[0]):
        for w in range(len(rule_selections)):
            if application_option == 0 or application_option == 3:
                try:
                    tup = (rule_selections[w][b][0], variable_selections[w][b][0])
                except:
                    tup = (rule_selections[w][b], variable_selections[w][b])
            elif application_option == 1:
                y = rule_selections[w][b]

                r1 = y[0] % num_rules
                v1 = y[0] % num_blocks
                r2  = y[1] % num_rules
                v2 = y[1] % num_blocks
                tup = (r1, v1, r2, v2)
            if tup not in variable_rules:
                variable_rules[tup] = 1
            else:
                variable_rules[tup] += 1
    return variable_rules