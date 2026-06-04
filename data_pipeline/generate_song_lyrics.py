import json

lyrics = {
    "01": "edelweiss edelweiss every morning you greet me small and white clean and bright you look happy to meet me blossom of snow may you bloom and grow bloom and grow forever edelweiss edelweiss bless my homeland forever",
    "02": "doe a deer a female deer ray a drop of golden sun me a name i call myself far a long long way to run sew a needle pulling thread la a note to follow sew tea a drink with jam and bread that will bring us back to do",
    "03": "dashing through the snow in a one horse open sleigh o'er the fields we go laughing all the way bells on bobtails ring making spirits bright what fun it is to ride and sing a sleighing song tonight oh jingle bells jingle bells jingle all the way oh what fun it is to ride in a one horse open sleigh",
    "04": "silent night holy night all is calm all is bright round yon virgin mother and child holy infant so tender and mild sleep in heavenly peace sleep in heavenly peace",
    "05": "it's late in the evening she's wondering what clothes to wear she puts on her make-up and brushes her long blonde hair and then she asks me do i look all right and i say yes you look wonderful tonight",
    "06": "moon river wider than a mile i'm crossing you in style some day oh dream maker you heart breaker wherever you're going i'm going your way two drifters off to see the world there's such a lot of world to see",
    "07": "listen to the rhythm of the falling rain telling me just what a fool i've been i wish that it would go and let me cry in vain and let me be alone again the only girl i care about has gone away looking for a brand new start but little does she know that when she left that day along with her she took my heart",
    "08": "i have a dream a song to sing to help me cope with anything if you see the wonder of a fairy tale you can take the future even if you fail i believe in angels something good in everything i see i believe in angels when i know the time is right for me i'll cross the stream i have a dream",
    "09": "love me tender love me sweet never let me go you have made my life complete and i love you so love me tender love me true all my dreams fulfill for my darling i love you and i always will",
    "10": "twinkle twinkle little star how i wonder what you are up above the world so high like a diamond in the sky twinkle twinkle little star how i wonder what you are",
    "11": "you are my sunshine my only sunshine you make me happy when skies are grey you'll never know dear how much i love you please don't take my sunshine away",
    "12": "greatness as you smallest as me you show me what is deep as sea a little love little kiss a little hug little gift all of little something these are our memories you make me cry make me smile make me feel that love is true you always stand by my side i don't want to say goodbye",
    "13": "love in your eyes sitting silent by my side going on holding hands walking through the nights hold me up hold me tight lift me up to touch the sky teaching me to love with heart helping me open my mind i can fly i'm proud that i can fly to give the best of mine",
    "14": "i'm sitting here in a boring room it's just another rainy sunday afternoon i'm wasting my time i got nothing to do i'm hanging around i'm waiting for you but nothing ever happens and i wonder i'm driving around in my car i'm driving too fast i'm driving too far i'd like to change my point of view",
    "15": "there's a calm surrender to the rush of day when the heat of a rolling wind can be turned away an enchanted moment and it sees me through it's enough for this restless warrior just to be with you and can you feel the love tonight it is where we are",
    "16": "i can't believe i'm standing here been waiting for so many years and today i found the queen to reign my heart you changed my life so patiently and turned it into something good and real i feel just like i felt in all my dreams there are questions hard to answer can't you see",
    "17": "goodbye to you my trusted friend we've known each other since we were nine or ten together we've climbed hills and trees learned of love and abc's skinned our hearts and skinned our knees goodbye my friend it's hard to die when all the birds are singing in the sky",
    "18": "i'm just a little bit caught in the middle life is a maze and love is a riddle i don't know where to go i can't do it alone i've tried and i don't know why slow it down make it stop or else my heart is going to pop 'cause it's too much yeah it's a lot to be something i'm not",
    "19": "some say love it is a river that drowns the tender reed some say love it is a razor that leaves your soul to bleed some say love it is a hunger an endless aching need i say love it is a flower and you its only seed",
    "20": "oceans apart day after day and i slowly go insane i hear your voice on the line but it doesn't stop the pain if i see you next to never how can we say forever wherever you go whatever you do i will be right here waiting for you whatever it takes or how my heart breaks i will be right here waiting for you",
}

with open("data/raw/nus48e/lyrics.json", "w") as f:
    json.dump(lyrics, f, indent=2)
print("Saved lyrics.json")
