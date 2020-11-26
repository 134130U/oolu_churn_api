select row_to_json(t) from (select total_amount, day_disabled, expected_total_amount, created_at::date,
            total_payed, status, cutoff_days, prev_payment::date
	   from account_analytics where status != 4 limit 10)t
	   limit 10