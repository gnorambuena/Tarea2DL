#include <bits/stdc++.h>
using namespace std;

bool play(vector<queue<int>> &T)
{
	int hand = T[12].front(); T[12].pop();
	int count = 0;
	while(count < 4)
	{
		T[hand].push(hand);
		int new_hand = T[hand].front(); T[hand].pop();
		if(new_hand == hand) count++;
		else count = 0;
		hand = new_hand;
	}

	for(int k = 0; k < 13; ++k)
		while(!T[k].empty())
		{
			int t = T[k].top(); T[k].pop();
			if(t != k) return false;
		}

	return true;
}

int main()
{
	ios::sync_with_stdio(0); cin.tie(0);

	map<char,int> M = {{'J',0},{'T',9},{'A',10},{'Q',11},{'K',12}};
	for(char d = '2'; d <= '9'; ++d) M[d] = d-'1';

	while(true)
	{
		int A[52];
		char c; cin >> c; if(c == '0') break;
		A[0] = M[c];

		for(int i = 1; i < 52; ++i)
		{
			char c; cin >> c;
			A[i] = M[c];
		}

		int ans = 0;
		for(int k = 0; k < 52; ++k)
		{
			vector<queue<int>> T(13);
			for(int d = 0; d < 52; ++d)
				T[d/4].push(A[(i+d)%52]);
			ans += play(T);
		}

		cout << ans << '\n';
	}

	return 0;
}