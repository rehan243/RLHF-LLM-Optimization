create or replace view potential_fraud_detection as
select 
    transactions.user_id,
    transactions.transaction_id,
    transactions.amount,
    transactions.timestamp,
    users.account_age,
    case 
        when transactions.amount > 1000 then 'high_value'
        when transactions.amount between 500 and 1000 then 'medium_value'
        else 'low_value'
    end as transaction_value_category,
    count(transactions.transaction_id) over (partition by transactions.user_id order by transactions.timestamp range between interval '30 days' preceding and current row) as recent_transaction_count
from 
    transactions
join 
    users on transactions.user_id = users.id
where 
    transactions.status = 'completed'
    and transactions.timestamp >= current_date - interval '90 days'
    and users.is_active = true;

-- TODO: might want to add more user attributes later for better insights
-- also consider indexing on user_id and timestamp for performance