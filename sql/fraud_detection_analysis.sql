create or replace view fraud_detection_analysis as
select 
    user_id,
    count(case when transaction_status = 'failed' then 1 end) as failed_transactions,
    count(case when transaction_status = 'success' then 1 end) as successful_transactions,
    sum(transaction_amount) as total_transaction_amount,
    avg(transaction_amount) as avg_transaction_amount,
    max(transaction_amount) as max_transaction_amount,
    min(transaction_amount) as min_transaction_amount
from 
    transactions
where 
    transaction_date >= current_date - interval '30 days'
group by 
    user_id
having 
    count(case when transaction_status = 'failed' then 1 end) > 3
order by 
    failed_transactions desc;

-- TODO: consider adding a threshold for total_transaction_amount if needed for better filtering
-- this view should help in identifying users with high fraud risk based on their transaction patterns