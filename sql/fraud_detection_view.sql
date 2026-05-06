create or replace view fraud_detection_view as
select 
    t.transaction_id,
    t.user_id,
    t.amount,
    t.transaction_date,
    u.account_age,
    case 
        when t.amount > 1000 then 'high_value'
        when t.amount between 500 and 1000 then 'medium_value'
        else 'low_value'
    end as transaction_value,
    case 
        when t.amount > 1000 and u.account_age < 30 then 'potential_fraud'
        when t.amount > 500 and u.account_age < 60 then 'review'
        else 'normal'
    end as fraud_risk
from 
    transactions t
join 
    users u on t.user_id = u.user_id
where 
    t.transaction_date >= current_date - interval '30 days'
    and t.transaction_status = 'completed'
order by 
    t.transaction_date desc;

-- TODO: consider indexing on transaction_date for better performance
-- this view aims to help in identifying potentially fraudulent activities based on transaction patterns