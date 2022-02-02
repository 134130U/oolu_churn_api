select row_to_json(t) from (select account_id,
 account_number,
 status,
 total_amount,
 invoce_model,
 created_at::date,
 total_payed,
 balance,
 group_name,
 prev_payment::date,
 prev_account_number,
 slug,
 cutoff_days,
 day_disabled,
 written_off,
 expected_total_amount
 
	   from account_analytics where status != 4)t
	   