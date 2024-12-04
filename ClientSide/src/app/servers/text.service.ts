import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class TextService {

  constructor(private http:HttpClient) { }
  translate(str:string) :Observable<String>
  {
    
   const url=' http://127.0.0.1:5000/translate'
   const params = { st: str };

   return this.http.get<String>(`${url}`,{params})
   
  }
}
