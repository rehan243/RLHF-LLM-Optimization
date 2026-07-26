create or replace view voice_analysis as
select 
    v.user_id,
    v.session_id,
    v.transcription,
    v.sentiment_score,
    v.duration,
    case 
        when v.sentiment_score > 0.5 then 'positive'
        when v.sentiment_score < -0.5 then 'negative'
        else 'neutral'
    end as sentiment_category,
    count(c.call_id) as call_count
from 
    voice_data v
left join 
    call_data c on v.session_id = c.session_id
group by 
    v.user_id, v.session_id, v.transcription, v.sentiment_score, v.duration
order by 
    v.user_id, call_count desc;

-- TODO: consider adding filters for specific date ranges in the future 
-- for more targeted analysis