def getLastTimestamp(string):
    last_timestamp=string.split(':')

    hour=int(last_timestamp[0])
    min=int(last_timestamp[1])
    sec=int(last_timestamp[2])

    sec+=1
    if(sec>=60):
        sec=0
        min+=1
        if(min>=60):
            min=0
            hour+=1
            if (hour>=24):
                hour=0

    hour=str(hour).zfill(2)
    min=str(min).zfill(2)
    sec=str(sec).zfill(2)

    end=hour+":"+min+":"+sec
    return end